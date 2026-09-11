//! Safe standalone mode-averaged facade over the existing resident CPU engine.
use crate::ionization::{AdkIonizationRate, CubicSplineLUT, PptIonizationRate, SplineSegment};
use crate::native::{CpuNativeSim, NativeBackend, NativeStepResult};
use crate::transforms::FftScratch;
use num_complex::Complex64 as C;

pub struct ModeAverageConfig {
    pub linop: Vec<C>,
    pub is_real: bool,
    pub prefactor: Vec<C>,
    pub time_window: Vec<f64>,
    pub filter_time: Vec<f64>,
    pub filter_frequency: Vec<f64>,
    /// Coefficient before the resident envelope's 3/4 averaging factor.
    pub kerr: f64,
    pub amplitude_scale: f64,
    /// Unpadded causal impulse samples, time step, and density.
    pub raman: Option<(Vec<f64>, f64, f64)>,
    pub plasma: Option<PlasmaConfig>,
}

pub enum IonizationConfig {
    Adk(AdkIonizationRate),
    Ppt {
        field: Vec<f64>,
        log_rate: Vec<f64>,
        derivative: Vec<f64>,
    },
}

pub struct PlasmaConfig {
    pub rate: IonizationConfig,
    pub ionpot: f64,
    pub e_ratio: f64,
    pub preionfrac: f64,
    pub dt: f64,
    pub density: f64,
}

enum OwnedIonization {
    Adk(Box<AdkIonizationRate>),
    Ppt(Box<PptIonizationRate>),
}

impl IonizationConfig {
    fn into_owned(self) -> Result<OwnedIonization, &'static str> {
        match self {
            Self::Adk(rate) => {
                if ![
                    rate.occupancy,
                    rate.omega_p,
                    rate.cn_sq,
                    rate.nstar,
                    rate.omega_t_prefac,
                    rate.thr,
                    rate.avfac,
                ]
                .iter()
                .all(|x| x.is_finite() && *x > 0.)
                {
                    return Err("invalid resident ADK coefficients or nonpositive threshold");
                }
                Ok(OwnedIonization::Adk(Box::new(rate)))
            }
            Self::Ppt {
                field,
                log_rate,
                derivative,
            } => {
                let n = field.len();
                if !(4..=1 << 24).contains(&n)
                    || log_rate.len() != n
                    || derivative.len() != n
                    || !finite(&field)
                    || !finite(&log_rate)
                    || !finite(&derivative)
                    || field[0] <= 0.
                    || field.windows(2).any(|w| w[1] <= w[0])
                {
                    return Err("invalid resident PPT samples");
                }
                let mut segments = Vec::with_capacity(n);
                for i in 0..n - 1 {
                    let h = field[i + 1] - field[i];
                    let delta = log_rate[i + 1] - log_rate[i];
                    let d0 = derivative[i];
                    let d1 = derivative[i + 1];
                    let b = d0 / h;
                    let c_numerator = 3. * delta - 2. * d0 - d1;
                    let d_numerator = 2. * (-delta) + d0 + d1;
                    let c = (c_numerator / h) / h;
                    let d = ((d_numerator / h) / h) / h;
                    if ![b, c, d].iter().all(|x| x.is_finite())
                        || [(b, d0), (c, c_numerator), (d, d_numerator)]
                            .iter()
                            .any(|(c, n)| *c == 0. && *n != 0.)
                    {
                        return Err("resident PPT polynomial coefficients exceed finite range");
                    }
                    segments.push(SplineSegment {
                        x: field[i],
                        a: log_rate[i],
                        b,
                        c,
                        d,
                    });
                }
                segments.push(SplineSegment {
                    x: field[n - 1],
                    a: log_rate[n - 1],
                    b: 0.,
                    c: 0.,
                    d: 0.,
                });
                Ok(OwnedIonization::Ppt(Box::new(PptIonizationRate {
                    spline_lut: CubicSplineLUT {
                        segments,
                        x_min: field[0],
                        x_max: field[n - 1],
                    },
                    e_min: field[0],
                    e_max: field[n - 1],
                    strict: false,
                })))
            }
        }
    }
}

pub struct ResidentModeAverage {
    // Drop the CPU engine before the owned rate whose pointer it borrows.
    sim: CpuNativeSim,
    _plasma_rate: Option<OwnedIonization>,
    filter_time: Vec<f64>,
    filter_frequency: Vec<f64>,
    time: Vec<C>,
    real_time: Vec<f64>,
    spectrum: Vec<C>,
    scratch: FftScratch,
}

fn finite_c(values: &[C]) -> bool {
    values.iter().all(|v| v.re.is_finite() && v.im.is_finite())
}
fn finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

impl ResidentModeAverage {
    pub fn new(config: ModeAverageConfig) -> Result<Self, &'static str> {
        let n = config.linop.len();
        let no = config.time_window.len();
        let nt = config.filter_time.len();
        let is_real = config.is_real;
        if n < 2
            || nt < 2
            || nt % 2 != 0
            || n != if is_real { nt / 2 + 1 } else { nt }
            || no < nt
            || no % 2 != 0
            || no > (1 << 23)
            || config.prefactor.len() != n
            || config.filter_frequency.len() != n
            || !finite_c(&config.linop)
            || !finite_c(&config.prefactor)
            || !finite(&config.time_window)
            || !finite(&config.filter_time)
            || !finite(&config.filter_frequency)
            || !config.kerr.is_finite()
            || !config.amplitude_scale.is_finite()
            || config.amplitude_scale <= 0.0
        {
            return Err("invalid resident mode-averaged configuration");
        }
        if let Some((h, dt, rho)) = &config.raman {
            if h.len() != no || !finite(h) || !dt.is_finite() || *dt <= 0.0 || !rho.is_finite() {
                return Err("invalid resident Raman samples");
            }
        }
        let plasma = match config.plasma {
            Some(p) => {
                if !is_real
                    || ![p.ionpot, p.e_ratio, p.dt]
                        .iter()
                        .all(|x| x.is_finite() && *x > 0.)
                    || !p.density.is_finite()
                    || p.density < 0.
                    || !p.preionfrac.is_finite()
                    || !(0.0..=1.0).contains(&p.preionfrac)
                {
                    return Err("invalid resident plasma configuration");
                }
                Some((
                    p.rate.into_owned()?,
                    p.ionpot,
                    p.e_ratio,
                    p.preionfrac,
                    p.dt,
                    p.density,
                ))
            }
            None => None,
        };
        let mut sim = CpuNativeSim::new_portable(&config.linop, nt, no, is_real)?;
        let pre_re: Vec<_> = config.prefactor.iter().map(|v| v.re).collect();
        let pre_im: Vec<_> = config.prefactor.iter().map(|v| v.im).collect();
        let ones = vec![1.0; n];
        let active = vec![1u8; n];
        // All slices have their validated exact lengths and the setter copies them.
        let status = unsafe {
            sim.set_mode_avg_params(
                nt,
                no,
                config.time_window.as_ptr(),
                ones.as_ptr(),
                active.as_ptr(),
                pre_re.as_ptr(),
                pre_im.as_ptr(),
                ones.as_ptr(),
                config.kerr,
                config.amplitude_scale,
                1.0,
            )
        };
        if status != 0 {
            return Err("resident mode-averaged setup failed");
        }
        if let Some((mut h, dt, rho)) = config.raman {
            h.resize(2 * no, 0.0);
            sim.configure_raman_samples(h, dt, rho);
        }
        let plasma_rate = match plasma {
            Some((rate, ionpot, e_ratio, preionfrac, dt, density)) => {
                // The box is retained by this facade and never moved out or replaced.
                // Its pointee remains stable when the containing facade is moved.
                let status = unsafe {
                    match &rate {
                        OwnedIonization::Adk(rate) => sim.set_plasma_params_adk(
                            &**rate, ionpot, e_ratio, preionfrac, dt, density,
                        ),
                        OwnedIonization::Ppt(rate) => {
                            sim.set_plasma_params(&**rate, ionpot, e_ratio, preionfrac, dt, density)
                        }
                    }
                };
                if status != 0 {
                    return Err("resident plasma setup failed");
                }
                Some(rate)
            }
            None => None,
        };
        let mut scratch = FftScratch::default();
        if is_real {
            sim.fft_r2c.as_ref().unwrap().prepare(&mut scratch);
        } else {
            sim.fft_c2c.as_ref().unwrap().prepare(&mut scratch);
        }
        Ok(Self {
            sim,
            _plasma_rate: plasma_rate,
            filter_time: config.filter_time,
            filter_frequency: config.filter_frequency,
            time: vec![C::default(); if is_real { 0 } else { nt }],
            real_time: vec![0.0; if is_real { nt } else { 0 }],
            spectrum: vec![C::default(); n],
            scratch,
        })
    }

    pub fn field(&self) -> &[C] {
        &self.sim.field
    }
    pub fn stages(&self) -> &[Vec<C>] {
        &self.sim.ks
    }
    pub fn initialize(&mut self, field: &[C]) -> Result<(), &'static str> {
        if field.len() != self.sim.n || !finite_c(field) {
            return Err("invalid resident initial field");
        }
        // Exact validated read length; no borrowed pointer is retained.
        if unsafe { self.sim.set_field(field.as_ptr().cast(), field.len()) } != 0 {
            return Err("resident initialization failed");
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attempt(
        &mut self,
        output: &mut [C],
        t: f64,
        tn: f64,
        dt: f64,
        rtol: f64,
        atol: f64,
        safety: f64,
        max_dt: f64,
        min_dt: f64,
        errlast: f64,
        fifth: bool,
    ) -> Result<NativeStepResult, &'static str> {
        if output.len() != self.sim.n
            || ![t, tn, errlast].iter().all(|v| v.is_finite())
            || ![dt, rtol, atol, safety, max_dt, min_dt]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
            || min_dt > max_dt
            || safety > 1.0
            || tn + dt <= tn
            || !(tn + dt).is_finite()
        {
            return Err("invalid resident attempt controls");
        }
        let mut result = NativeStepResult {
            ok: 0,
            dt: 0.0,
            t: 0.0,
            tn: 0.0,
            dtn: 0.0,
            err: 0.0,
            errlast: 0.0,
        };
        // Output and result exclusively borrowed for this synchronous call.
        let status = unsafe {
            self.sim.step(
                output.as_mut_ptr(),
                t,
                tn,
                dt,
                rtol,
                atol,
                safety,
                max_dt,
                min_dt,
                errlast,
                i32::from(fifth),
                &mut result,
            )
        };
        if status != 0 {
            return Err("resident attempt failed");
        }
        Ok(result)
    }
    pub fn extra_stages(&mut self, left: &[C], t: f64, dt: f64) -> Result<(), &'static str> {
        if left.len() != self.sim.n
            || !finite_c(left)
            || !t.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
        {
            return Err("invalid resident dense interval");
        }
        if unsafe {
            self.sim
                .compute_extra_stages(left.as_ptr().cast(), left.len(), t, dt)
        } != 0
        {
            return Err("resident dense stages failed");
        }
        Ok(())
    }
    pub fn filter(&mut self) -> Result<(), &'static str> {
        for ((out, input), win) in self
            .spectrum
            .iter_mut()
            .zip(&self.sim.field)
            .zip(&self.filter_frequency)
        {
            *out = *input * win;
        }
        if self.sim.is_real {
            let fft = self.sim.fft_r2c.as_ref().unwrap();
            fft.inverse(&mut self.spectrum, &mut self.real_time, &mut self.scratch);
            for (value, window) in self.real_time.iter_mut().zip(&self.filter_time) {
                *value *= self.sim.fft_norm * window;
            }
            fft.forward(&mut self.real_time, &mut self.spectrum, &mut self.scratch);
        } else {
            let fft = self.sim.fft_c2c.as_ref().unwrap();
            fft.inverse(&mut self.spectrum, &mut self.time, &mut self.scratch);
            for (value, window) in self.time.iter_mut().zip(&self.filter_time) {
                *value *= self.sim.fft_norm * window;
            }
            fft.forward(&mut self.time, &mut self.spectrum, &mut self.scratch);
        }
        if !finite_c(&self.spectrum) {
            return Err("nonfinite resident filtered field");
        }
        // Match Julia resync: retain the completed interval's FSAL derivative.
        self.sim.field.copy_from_slice(&self.spectrum);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adk() -> IonizationConfig {
        IonizationConfig::Adk(AdkIonizationRate {
            occupancy: 2.,
            omega_p: 1.,
            cn_sq: 1.,
            nstar: 1.,
            omega_t_prefac: 1.,
            thr: 1e-10,
            avfac: 1.,
        })
    }
    fn ppt() -> IonizationConfig {
        IonizationConfig::Ppt {
            field: vec![1., 2., 4., 5.],
            log_rate: vec![0., 1., 2., 3.],
            derivative: vec![1., 2., 3., 4.],
        }
    }
    fn config(rate: IonizationConfig) -> ModeAverageConfig {
        ModeAverageConfig {
            linop: vec![C::new(-0.01, 0.); 9],
            is_real: true,
            prefactor: vec![C::new(1., 0.); 9],
            time_window: vec![1.; 16],
            filter_time: vec![1.; 16],
            filter_frequency: vec![1.; 9],
            kerr: 0.3,
            amplitude_scale: 1.,
            raman: None,
            plasma: Some(PlasmaConfig {
                rate,
                ionpot: 1.,
                e_ratio: 1.,
                preionfrac: 0.,
                dt: 0.02,
                density: 1.,
            }),
        }
    }

    #[test]
    fn owned_plasma_survives_facade_moves_and_destruction() {
        for rate in [adk(), ppt()] {
            let mut sim = ResidentModeAverage::new(config(rate)).unwrap();
            let mut input = vec![C::new(0., 0.); 9];
            input[0] = C::new(32., 0.);
            sim.initialize(&input).unwrap();
            let reference = sim.stages()[0].clone();
            let pointer = if sim.sim.plasma_is_adk {
                sim.sim.plasma_adk_ptr.cast::<()>()
            } else {
                sim.sim.plasma_ion_ptr.cast::<()>()
            };
            let mut moved = vec![sim];
            for _ in 0..32 {
                moved.push(ResidentModeAverage::new(config(ppt())).unwrap());
            }
            let mut sim = moved.swap_remove(0);
            drop(moved);
            let after = if sim.sim.plasma_is_adk {
                sim.sim.plasma_adk_ptr.cast::<()>()
            } else {
                sim.sim.plasma_ion_ptr.cast::<()>()
            };
            assert_eq!(pointer, after);
            sim.initialize(&input).unwrap();
            assert_eq!(sim.stages()[0], reference);
            assert!(reference.iter().any(|z| z.norm() > 0.));
            let mut output = input.clone();
            assert_eq!(
                sim.attempt(
                    &mut output,
                    0.,
                    0.,
                    0.001,
                    1e-6,
                    1e-10,
                    0.9,
                    0.001,
                    0.001,
                    0.,
                    true
                )
                .unwrap()
                .ok,
                1
            );
            assert!(finite_c(&output));
            sim.filter().unwrap();
        }
    }

    #[test]
    fn normalized_ppt_segments_preserve_nonuniform_samples_and_clamping() {
        let OwnedIonization::Ppt(rate) = ppt().into_owned().unwrap() else {
            panic!()
        };
        for (i, (x, y)) in [(1., 0.), (2., 1.), (4., 2.), (5., 3.)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(rate.rate(x).unwrap(), f64::exp(y));
            assert_eq!(rate.spline_lut.segments[i].x, x);
        }
        let t = 0.3;
        let delta = 1.;
        let d0 = 2.;
        let d1 = 3.;
        let log =
            1. + d0 * t + (3. * delta - 2. * d0 - d1) * t * t + (-2. * delta + d0 + d1) * t * t * t;
        assert!((rate.rate(2. + 2. * t).unwrap() / f64::exp(log) - 1.).abs() < 1e-13);
        assert_eq!(rate.rate(0.).unwrap(), 0.);
        assert_eq!(rate.rate(-10.).unwrap(), rate.rate(5.).unwrap());
    }

    #[test]
    fn invalid_owned_plasma_is_rejected_before_engine_setup() {
        for index in 0..5 {
            let mut c = config(adk());
            let p = c.plasma.as_mut().unwrap();
            match index {
                0 => p.ionpot = 0.,
                1 => p.e_ratio = f64::NAN,
                2 => p.preionfrac = 1.1,
                3 => p.dt = 0.,
                _ => p.density = -1.,
            };
            assert!(ResidentModeAverage::new(c).is_err());
        }
        let IonizationConfig::Adk(mut rate) = adk() else {
            panic!()
        };
        rate.thr = 0.;
        assert!(IonizationConfig::Adk(rate).into_owned().is_err());
        for fields in [
            vec![1., 2., 3.],
            vec![1., 2., 2., 4.],
            vec![1., 2., 3., f64::NAN],
        ] {
            assert!(
                IonizationConfig::Ppt {
                    field: fields,
                    log_rate: vec![0.; 4],
                    derivative: vec![1.; 4]
                }
                .into_owned()
                .is_err()
            );
        }
        assert!(
            IonizationConfig::Ppt {
                field: vec![1e-200, 2e-200, 3e-200, 4e-200],
                log_rate: vec![0., 1., 3., 6.],
                derivative: vec![1.; 4]
            }
            .into_owned()
            .is_err()
        );
        assert!(
            IonizationConfig::Ppt {
                field: vec![1e200, 2e200, 3e200, 4e200],
                log_rate: vec![0., 1., 3., 6.],
                derivative: vec![1.; 4]
            }
            .into_owned()
            .is_err()
        );
    }
}
