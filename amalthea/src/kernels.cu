#include <math.h>

#define MAX_OSCILLATORS 32

struct PrecomputedStepCoeffs {
    double a11, a12, a21, a22;
    double b0_1, b0_2;
    double b1_1, b1_2;
};

struct SplineSegment {
    double x;
    double a;
    double b;
    double c;
    double d;
};

extern "C" __global__ void raman_ade_kernel(
    const double* intensity,
    double* raman_polarization,
    const PrecomputedStepCoeffs* coeffs,
    int num_oscillators,
    int n_t,
    int n_series
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_series) return;
    
    double q_states[MAX_OSCILLATORS];
    double dq_states[MAX_OSCILLATORS];
    
    int num_osc = num_oscillators > MAX_OSCILLATORS ? MAX_OSCILLATORS : num_oscillators;
    for (int i = 0; i < num_osc; i++) {
        q_states[i] = 0.0;
        dq_states[i] = 0.0;
    }
    
    int offset = s * n_t;
    raman_polarization[offset] = 0.0;
    
    for (int n = 0; n < n_t - 1; n++) {
        double i_n = intensity[offset + n];
        double i_np1 = intensity[offset + n + 1];
        
        double total_q = 0.0;
        for (int i = 0; i < num_osc; i++) {
            PrecomputedStepCoeffs c = coeffs[i];
            double q = q_states[i];
            double dq = dq_states[i];
            
            double q_new = c.a11 * q + c.a12 * dq + c.b0_1 * i_n + c.b1_1 * i_np1;
            double dq_new = c.a21 * q + c.a22 * dq + c.b0_2 * i_n + c.b1_2 * i_np1;
            
            q_states[i] = q_new;
            dq_states[i] = dq_new;
            
            total_q += q_new;
        }
        raman_polarization[offset + n + 1] = total_q;
    }
}

extern "C" __global__ void ppt_ionization_kernel(
    const double* fields,
    double* rates,
    const SplineSegment* segments,
    double e_min,
    double e_max,
    int num_segments,
    int N,
    int* err_code,
    int strict
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    double abs_e = fabs(fields[idx]);
    if (abs_e < e_min) {
        rates[idx] = 0.0;
        return;
    }
    
    if (abs_e > e_max) {
        if (strict) {
            rates[idx] = -1.0;
            atomicExch(err_code, 1);
            return;
        } else {
            abs_e = e_max;
        }
    }
    
    int low = 0;
    int high = num_segments - 1;
    while (low < high) {
        int mid = (low + high + 1) / 2;
        if (segments[mid].x <= abs_e) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    
    const SplineSegment seg = segments[low];
    double dx = abs_e - seg.x;
    double ln_rate = seg.a + dx * (seg.b + dx * (seg.c + dx * seg.d));
    rates[idx] = exp(ln_rate);
}
#include <cuComplex.h>

extern "C" __global__ void apply_prop_kernel(cuDoubleComplex* y, const cuDoubleComplex* linop, int n, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    double real_part = linop[idx].x * dt;
    double imag_part = linop[idx].y * dt;
    
    double exp_val = exp(real_part);
    double cos_val = cos(imag_part);
    double sin_val = sin(imag_part);
    
    cuDoubleComplex prop = make_cuDoubleComplex(exp_val * cos_val, exp_val * sin_val);
    y[idx] = cuCmul(y[idx], prop);
}

// Fused RK45 stage accumulation
extern "C" __global__ void rk45_accumulate_stage_kernel(
    cuDoubleComplex* ystage,
    const cuDoubleComplex* field,
    const cuDoubleComplex* k0,
    const cuDoubleComplex* k1,
    const cuDoubleComplex* k2,
    const cuDoubleComplex* k3,
    const cuDoubleComplex* k4,
    const cuDoubleComplex* k5,
    const cuDoubleComplex* k6,
    double b0, double b1, double b2, double b3, double b4, double b5, double b6,
    int n, double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    cuDoubleComplex f = field[idx];
    
    double re = f.x;
    double im = f.y;
    
    if (b0 != 0.0) { re += dt * b0 * k0[idx].x; im += dt * b0 * k0[idx].y; }
    if (b1 != 0.0) { re += dt * b1 * k1[idx].x; im += dt * b1 * k1[idx].y; }
    if (b2 != 0.0) { re += dt * b2 * k2[idx].x; im += dt * b2 * k2[idx].y; }
    if (b3 != 0.0) { re += dt * b3 * k3[idx].x; im += dt * b3 * k3[idx].y; }
    if (b4 != 0.0) { re += dt * b4 * k4[idx].x; im += dt * b4 * k4[idx].y; }
    if (b5 != 0.0) { re += dt * b5 * k5[idx].x; im += dt * b5 * k5[idx].y; }
    if (b6 != 0.0) { re += dt * b6 * k6[idx].x; im += dt * b6 * k6[idx].y; }
    
    ystage[idx] = make_cuDoubleComplex(re, im);
}

// Error estimation kernel
extern "C" __global__ void rk45_accumulate_error_kernel(
    cuDoubleComplex* yerr,
    const cuDoubleComplex* k0,
    const cuDoubleComplex* k1,
    const cuDoubleComplex* k2,
    const cuDoubleComplex* k3,
    const cuDoubleComplex* k4,
    const cuDoubleComplex* k5,
    const cuDoubleComplex* k6,
    double e0, double e1, double e2, double e3, double e4, double e5, double e6,
    int n, double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    double re = 0.0;
    double im = 0.0;
    
    if (e0 != 0.0) { re += dt * e0 * k0[idx].x; im += dt * e0 * k0[idx].y; }
    if (e1 != 0.0) { re += dt * e1 * k1[idx].x; im += dt * e1 * k1[idx].y; }
    if (e2 != 0.0) { re += dt * e2 * k2[idx].x; im += dt * e2 * k2[idx].y; }
    if (e3 != 0.0) { re += dt * e3 * k3[idx].x; im += dt * e3 * k3[idx].y; }
    if (e4 != 0.0) { re += dt * e4 * k4[idx].x; im += dt * e4 * k4[idx].y; }
    if (e5 != 0.0) { re += dt * e5 * k5[idx].x; im += dt * e5 * k5[idx].y; }
    if (e6 != 0.0) { re += dt * e6 * k6[idx].x; im += dt * e6 * k6[idx].y; }
    
    yerr[idx] = make_cuDoubleComplex(re, im);
}

// Weaknorm reduction (part 1: element-wise square)
extern "C" __global__ void weaknorm_elem_kernel(
    const cuDoubleComplex* yerr,
    const cuDoubleComplex* y0,
    const cuDoubleComplex* y1,
    double rtol, double atol,
    double* out_sq,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    double err_re = yerr[idx].x;
    double err_im = yerr[idx].y;
    double y0_re = y0[idx].x;
    double y0_im = y0[idx].y;
    double y1_re = y1[idx].x;
    double y1_im = y1[idx].y;
    
    double sq_y0 = y0_re * y0_re + y0_im * y0_im;
    double sq_y1 = y1_re * y1_re + y1_im * y1_im;
    double max_sq = sq_y0 > sq_y1 ? sq_y0 : sq_y1;
    
    double tol = atol + rtol * sqrt(max_sq);
    double err_sq = (err_re * err_re + err_im * err_im) / (tol * tol);
    
    out_sq[idx] = err_sq;
}

// Mode average real kerr. Deliberately does NOT apply `towin` (unlike its
// pre-plasma version) — matches native.rs's CpuNativeSim::rhs_mode_avg_real,
// where the time-domain window is applied once to the *combined* Pto
// (Kerr + plasma [+ Raman]), not to Kerr alone. See apply_time_window_kernel
// below, always run after every additive Pto contribution.
extern "C" __global__ void rhs_mode_avg_real_kernel(
    double* pto,
    const double* eto,
    double kerr_fac,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;

    double e = eto[idx];
    pto[idx] = kerr_fac * e * e * e;
}

// Time-domain window apodization: Pto *= towin. Split out from
// rhs_mode_avg_real_kernel so it can run once, after Kerr AND plasma have
// both contributed to `pto` — reproduces native.rs's Step 4 (applied to the
// combined Pto), not a per-response window.
extern "C" __global__ void apply_time_window_kernel(
    double* pto,
    const double* towin,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    pto[idx] *= towin[idx];
}

// Plasma ionization fraction: fused cumtrapz(rate)*dt + rho transform.
// Reproduces native.rs's apply_plasma_real steps 2-3 (Maths.cumtrapz! then
// `preionfrac + 1 - exp(-integral)`) in one single-thread sequential pass —
// cumtrapz is an inherently sequential prefix sum (each element depends on
// the previous), and n_time is small enough (~2^13-2^16) that a single
// thread looping over it is negligible next to this step's FFT/kernel-launch
// cost; not the work-efficient parallel prefix scan GPU.md's original design
// sketch mentions; a fine V1 tradeoff at this problem size; a parallel scan
// would matter at radial's much larger N=n_time*n_r plasma-state size, not
// implemented here.
extern "C" __global__ void plasma_fraction_kernel(
    const double* rate,
    double* fraction,
    double preionfrac,
    double dt,
    int n_time
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double acc = 0.0;
    fraction[0] = preionfrac + 1.0 - exp(-acc);
    for (int i = 1; i < n_time; i++) {
        acc += 0.5 * (rate[i - 1] + rate[i]) * dt;
        fraction[i] = preionfrac + 1.0 - exp(-acc);
    }
}

// Plasma phase: phase[i] = fraction[i] * e_ratio * eto[i] — elementwise,
// parallel (native.rs step 4).
extern "C" __global__ void plasma_phase_kernel(
    const double* fraction,
    const double* eto,
    double e_ratio,
    double* phase,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    phase[idx] = fraction[idx] * e_ratio * eto[idx];
}

// Free-electron current: fused cumtrapz(phase)*dt + ionization-loss-current
// add-in. Reproduces native.rs steps 5-6 (`cumtrapz(J, phase, dt)` then
// `J[i] += Ip*W[i]*(1-rho[i])/E[i]`) in one sequential pass, same rationale
// as plasma_fraction_kernel above.
extern "C" __global__ void plasma_current_kernel(
    const double* phase,
    const double* rate,
    const double* fraction,
    const double* eto,
    double ionpot,
    double dt,
    double* current,
    int n_time
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double acc = 0.0;
    for (int i = 0; i < n_time; i++) {
        if (i > 0) {
            acc += 0.5 * (phase[i - 1] + phase[i]) * dt;
        }
        double e = eto[i];
        double loss = (e != 0.0) ? ionpot * rate[i] * (1.0 - fraction[i]) / e : 0.0;
        current[i] = acc + loss;
    }
}

// Plasma polarisation: fused cumtrapz(current)*dt + accumulate into Pto.
// Reproduces native.rs steps 7-8 (`cumtrapz(P, J, dt)` then
// `pto[i] += density * P[i]`) in one sequential pass, same rationale as
// plasma_fraction_kernel above. Writes directly into the shared `pto`
// buffer that rhs_mode_avg_real_kernel already populated with the Kerr
// contribution — additive, matches native.rs's `self.pto[i] += ...`.
extern "C" __global__ void plasma_polarization_kernel(
    const double* current,
    double* pto,
    double density,
    double dt,
    int n_time
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double acc = 0.0;
    for (int i = 0; i < n_time; i++) {
        if (i > 0) {
            acc += 0.5 * (current[i - 1] + current[i]) * dt;
        }
        pto[i] += density * acc;
    }
}

// Mode average env kerr
extern "C" __global__ void rhs_mode_avg_env_kernel(
    cuDoubleComplex* poo,
    const cuDoubleComplex* eoo,
    const double* towin,
    double kerr_fac,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    
    cuDoubleComplex e = eoo[idx];
    double w = towin ? towin[idx] : 1.0;
    
    double mag_sq = e.x * e.x + e.y * e.y;
    double fac = w * kerr_fac * mag_sq;
    
    poo[idx] = make_cuDoubleComplex(fac * e.x, fac * e.y);
}

extern "C" __global__ void weaknorm_reduce_kernel(
    const double* in,
    double* out,
    int n
) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    double myMax = 0.0;
    if (i < n) myMax = in[i];
    if (i + blockDim.x < n) {
        double other = in[i + blockDim.x];
        if (other > myMax) myMax = other;
    }
    
    sdata[tid] = myMax;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
