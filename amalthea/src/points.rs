//! Safe reusable scalar/vector physical point batches for Python modal quadrature.
use crate::native::PortableModalPoint;
use crate::resident::{ModeAverageConfig, ResidentModeAverage};
use num_complex::Complex64 as C;

pub struct PointConfig {
    pub n_time: usize,
    pub components: usize,
    pub is_real: bool,
    pub time_window: Vec<f64>,
    pub prefactor: Vec<C>,
    /// Physical coefficient before envelope 3/4 averaging.
    pub kerr: f64,
    /// Unpadded, already density-scaled scalar impulse samples.
    pub raman: Option<Vec<f64>>,
    pub dt: f64,
}

enum Engine {
    Scalar(ResidentModeAverage),
    Vector(PortableModalPoint),
}

pub struct PointBatch {
    engine: Engine,
    width: usize,
}

impl PointBatch {
    pub fn new(config: PointConfig) -> Result<Self, &'static str> {
        let nt = config.n_time;
        let no = config.time_window.len();
        let n = config.prefactor.len();
        if nt < 2
            || nt > 1 << 23
            || nt % 2 != 0
            || no < nt
            || no > 1 << 23
            || no % 2 != 0
            || n != if config.is_real { nt / 2 + 1 } else { nt }
            || !(1..=2).contains(&config.components)
            || !config.time_window.iter().all(|v| v.is_finite())
            || !config
                .prefactor
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
            || !config.kerr.is_finite()
            || !config.dt.is_finite()
            || config.dt <= 0.0
        {
            return Err("invalid native point configuration");
        }
        if let Some(h) = &config.raman {
            if config.components != 1 || h.len() != no || !h.iter().all(|v| v.is_finite()) {
                return Err(
                    "native point Raman requires a finite scalar impulse on the full time grid",
                );
            }
        }
        let width = n
            .checked_mul(config.components)
            .ok_or("native point size overflow")?;
        let engine = if config.components == 1 {
            Engine::Scalar(ResidentModeAverage::new(ModeAverageConfig {
                linop: vec![C::default(); n],
                is_real: config.is_real,
                prefactor: config.prefactor,
                time_window: config.time_window,
                filter_time: vec![1.0; nt],
                filter_frequency: vec![1.0; n],
                kerr: config.kerr,
                amplitude_scale: 1.0,
                raman: config.raman.map(|h| (h, config.dt, 1.0)),
                plasma: None,
            })?)
        } else {
            Engine::Vector(PortableModalPoint::new(
                config.is_real,
                config.time_window,
                config.prefactor,
                config.kerr,
            ))
        };
        Ok(Self { engine, width })
    }

    /// Point-major, component-major spectra; no input storage is retained.
    pub fn evaluate(&mut self, fields: &[C], points: usize) -> Result<Vec<C>, &'static str> {
        let length = points
            .checked_mul(self.width)
            .ok_or("native point batch size overflow")?;
        if length > 1 << 24
            || fields.len() != length
            || !fields.iter().all(|v| v.re.is_finite() && v.im.is_finite())
        {
            return Err("invalid native point batch shape or values");
        }
        let mut result = Vec::with_capacity(length);
        for field in fields.chunks_exact(self.width) {
            let response = match &mut self.engine {
                Engine::Scalar(engine) => {
                    engine.initialize(field)?;
                    &engine.stages()[0]
                }
                Engine::Vector(engine) => engine.evaluate(field),
            };
            if !response
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
            {
                return Err("nonfinite native point polarization");
            }
            result.extend_from_slice(response);
        }
        Ok(result)
    }
}
