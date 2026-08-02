#include <math.h>
#include <cuComplex.h>
#include "cuda_raman_limits.h"

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
    
    // The Rust FFI setters reject counts outside the generated capacity
    // contract before this kernel is launched. Do not clamp here: dropping
    // oscillators silently changes the Raman polarization.
    double q_states[AMALTHEA_CUDA_RAMAN_MAX_OSCILLATORS];
    double dq_states[AMALTHEA_CUDA_RAMAN_MAX_OSCILLATORS];
    int num_osc = num_oscillators;
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

// Raman intensity/accumulation helpers used by the resident GPU stepper.
// The ADE recurrence itself remains in raman_ade_kernel so the CUDA and CPU
// paths share PrecomputedStepCoeffs exactly.
extern "C" __global__ void raman_intensity_real_kernel(
    const double* eto,
    double* intensity,
    int n,
    int thg
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double e = eto[idx];
    intensity[idx] = thg ? e * e : 0.0;
}

extern "C" __global__ void raman_hilbert_pack_kernel(
    const double* eto,
    cuDoubleComplex* a,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = make_cuDoubleComplex(eto[idx], 0.0);
}

extern "C" __global__ void raman_hilbert_filter_kernel(
    cuDoubleComplex* spectrum,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int half = n / 2;
    if (idx == 0 || (n % 2 == 0 && idx == half)) {
        // DC and (for even n) Nyquist remain single-weighted.
    } else if (idx <= half) {
        spectrum[idx].x *= 2.0;
        spectrum[idx].y *= 2.0;
    } else {
        spectrum[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

extern "C" __global__ void raman_hilbert_intensity_kernel(
    const cuDoubleComplex* analytic,
    double* intensity,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double re = analytic[idx].x;
    double im = analytic[idx].y;
    intensity[idx] = 0.5 * (re * re + im * im);
}

extern "C" __global__ void raman_intensity_env_kernel(
    const cuDoubleComplex* eto,
    double* intensity,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double re = eto[idx].x;
    double im = eto[idx].y;
    intensity[idx] = 0.5 * (re * re + im * im);
}

extern "C" __global__ void raman_accumulate_real_kernel(
    double* pto,
    const double* eto,
    const double* raman_polarization,
    double density,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    pto[idx] += density * eto[idx] * raman_polarization[idx];
}

extern "C" __global__ void raman_accumulate_env_kernel(
    cuDoubleComplex* pto,
    const cuDoubleComplex* eto,
    const double* raman_polarization,
    double density,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double factor = density * raman_polarization[idx];
    pto[idx].x += factor * eto[idx].x;
    pto[idx].y += factor * eto[idx].y;
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

// Closed-form ADK rate, matching `AdkIonizationRate::rate` exactly. The
// constants are precomputed on the Julia/CPU side and transferred verbatim;
// unlike PPT this kernel needs no lookup table or error channel.
extern "C" __global__ void adk_ionization_kernel(
    const double* fields,
    double* rates,
    double occupancy,
    double omega_p,
    double cn_sq,
    double nstar,
    double omega_t_prefac,
    double thr,
    double avfac,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double abs_e = fabs(fields[idx]);
    if (!isfinite(abs_e) || abs_e < thr) {
        rates[idx] = 0.0;
        return;
    }

    double x = 4.0 * omega_p / (omega_t_prefac * abs_e);
    double rate = occupancy * omega_p * cn_sq *
                  pow(x, 2.0 * nstar - 1.0) *
                  exp((-4.0 / 3.0) * omega_p / (omega_t_prefac * abs_e));
    if (avfac != 1.0) {
        rate *= avfac * sqrt(abs_e);
    }
    rates[idx] = rate;
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

// Weaknorm reduction (part 1): emit the three squared-magnitude arrays used
// by native.rs::weaknorm_c64. The tolerance is global, not an element-wise
// `(atol + rtol*max(abs(y0[i]),abs(y1[i])))` weight.
extern "C" __global__ void weaknorm_elem_kernel(
    const cuDoubleComplex* yerr,
    const cuDoubleComplex* y0,
    const cuDoubleComplex* y1,
    double* yerr_sq,
    double* y0_sq,
    double* y1_sq,
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
    
    yerr_sq[idx] = err_re * err_re + err_im * err_im;
    y0_sq[idx] = y0_re * y0_re + y0_im * y0_im;
    y1_sq[idx] = y1_re * y1_re + y1_im * y1_im;
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

// Step 1 (native.rs::rhs_mode_avg_real): zero-pad + scale the ODE-state
// spectrum (length n_spec) into the oversampled spectral buffer (length
// n_spec_over) that the inverse FFT expects, ahead of the nonlinear
// evaluation on the oversampled real-space grid. `in`/`out` may be the same
// length only when n_spec == n_spec_over (no oversampling); the general
// case is n_spec_over >= n_spec (BACKLOG.md S3 item 6's sizing fix).
extern "C" __global__ void expand_spectrum_kernel(
    const cuDoubleComplex* in,
    cuDoubleComplex* out,
    double scale_fwd,
    int n_spec,
    int n_spec_over
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spec_over) return;
    if (idx < n_spec) {
        cuDoubleComplex v = in[idx];
        out[idx] = make_cuDoubleComplex(v.x * scale_fwd, v.y * scale_fwd);
    } else {
        out[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

// Generic real-array scalar multiply. Used to fold native.rs's Step 1
// (cuFFT's unnormalized-inverse `1/n_time_over` factor) together with
// Step 2 (`1/(nlscale*sqrt_aeff)`) into a single pass over `eto`.
extern "C" __global__ void scale_real_kernel(
    double* a,
    double factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] *= factor;
}

// Step 5+6+7 (native.rs::rhs_mode_avg_real): crop the oversampled forward-FFT
// output (length n_spec_over) back to n_spec and scale by `scale_inv`
// (Step 5), then multiply by the precomputed `norm_pre_beta` (Step 6,
// `pre/beta*sqrt_aeff`, already folded to identity outside `sidx` on the
// host) and `owin` (Step 7, already folded to 1.0 outside `sidx`).
extern "C" __global__ void finalize_spectrum_kernel(
    const cuDoubleComplex* poo,
    cuDoubleComplex* ks_out,
    const cuDoubleComplex* norm_pre_beta,
    const double* owin,
    double scale_inv,
    int n_spec
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spec) return;

    double vre = poo[idx].x * scale_inv;
    double vim = poo[idx].y * scale_inv;

    cuDoubleComplex npb = norm_pre_beta[idx];
    double re = vre * npb.x - vim * npb.y;
    double im = vre * npb.y + vim * npb.x;

    double w = owin[idx];
    ks_out[idx] = make_cuDoubleComplex(re * w, im * w);
}

// Two-level trapezoidal prefix scan used by all three PPT cumulative
// integrals. Each 256-thread block performs a work-efficient Blelloch scan
// of q[0]=0, q[i]=0.5*(x[i-1]+x[i])*dt and records one block total. A tiny
// follow-up kernel scans those totals; the physics-specific finalizers below
// add the preceding-block offset in parallel.
extern "C" __global__ void plasma_scan_blocks_kernel(
    const double* input,
    double* local_prefix,
    double* block_sums,
    double dt,
    int n_time
) {
    extern __shared__ double temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double q = 0.0;
    if (idx > 0 && idx < n_time) {
        q = 0.5 * (input[idx - 1] + input[idx]) * dt;
    }
    temp[tid] = q;
    __syncthreads();

    // Upsweep: total lands in temp[blockDim.x-1].
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < blockDim.x) {
            temp[ai] += temp[ai - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = temp[blockDim.x - 1];
        temp[blockDim.x - 1] = 0.0;
    }
    __syncthreads();

    // Downsweep: convert the block scan to exclusive form.
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < blockDim.x) {
            double left = temp[ai - offset];
            temp[ai - offset] = temp[ai];
            temp[ai] += left;
        }
        __syncthreads();
    }

    if (idx < n_time) {
        local_prefix[idx] = temp[tid] + q;
    }
}

extern "C" __global__ void plasma_scan_block_sums_kernel(
    double* block_sums,
    int n_blocks
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double acc = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        acc += block_sums[i];
        block_sums[i] = acc;
    }
}

extern "C" __global__ void plasma_fraction_finalize_kernel(
    double* fraction,
    const double* block_sums,
    double preionfrac,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    double offset = blockIdx.x == 0 ? 0.0 : block_sums[blockIdx.x - 1];
    double acc = fraction[idx] + offset;
    fraction[idx] = preionfrac + 1.0 - exp(-acc);
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

// Finalize cumtrapz(phase)*dt with the ionization-loss-current add-in.
extern "C" __global__ void plasma_current_finalize_kernel(
    double* current,
    const double* block_sums,
    const double* rate,
    const double* fraction,
    const double* eto,
    double ionpot,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    double offset = blockIdx.x == 0 ? 0.0 : block_sums[blockIdx.x - 1];
    double e = eto[idx];
    double loss = (e != 0.0) ? ionpot * rate[idx] * (1.0 - fraction[idx]) / e : 0.0;
    current[idx] += offset + loss;
}

// Finalize cumtrapz(current)*dt and accumulate into the shared Kerr Pto.
extern "C" __global__ void plasma_polarization_finalize_kernel(
    const double* polarization_prefix,
    const double* block_sums,
    double* pto,
    double density,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    double offset = blockIdx.x == 0 ? 0.0 : block_sums[blockIdx.x - 1];
    pto[idx] += density * (polarization_prefix[idx] + offset);
}

// Mode average env kerr
extern "C" __global__ void rhs_mode_avg_env_kernel(
    cuDoubleComplex* pto,
    const cuDoubleComplex* eto,
    double kerr_fac,
    int n_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_time) return;
    
    cuDoubleComplex e = eto[idx];
    double mag_sq = e.x * e.x + e.y * e.y;
    double fac = 0.75 * kerr_fac * mag_sq;
    
    pto[idx] = make_cuDoubleComplex(fac * e.x, fac * e.y);
}

extern "C" __global__ void expand_spectrum_env_kernel(
    const cuDoubleComplex* in,
    cuDoubleComplex* out,
    double scale_fwd,
    int n_spec,
    int n_spec_over
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spec_over) return;
    int half = n_spec / 2;
    if (idx < half) {
        cuDoubleComplex v = in[idx];
        out[idx] = make_cuDoubleComplex(v.x * scale_fwd, v.y * scale_fwd);
    } else if (idx >= n_spec_over - half) {
        int src = n_spec - half + idx - (n_spec_over - half);
        cuDoubleComplex v = in[src];
        out[idx] = make_cuDoubleComplex(v.x * scale_fwd, v.y * scale_fwd);
    } else {
        out[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

extern "C" __global__ void scale_complex_kernel(
    cuDoubleComplex* a,
    double factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx].x *= factor;
    a[idx].y *= factor;
}

extern "C" __global__ void apply_time_window_complex_kernel(
    cuDoubleComplex* pto,
    const double* towin,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    pto[idx].x *= towin[idx];
    pto[idx].y *= towin[idx];
}

extern "C" __global__ void finalize_spectrum_env_kernel(
    const cuDoubleComplex* poo,
    cuDoubleComplex* ks_out,
    const cuDoubleComplex* norm_pre_beta,
    const double* owin,
    double scale_inv,
    int n_spec,
    int n_spec_over
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spec) return;
    int half = n_spec / 2;
    cuDoubleComplex v;
    if (idx < half) {
        v = poo[idx];
    } else if (idx >= n_spec - half) {
        int src = n_spec_over - half + idx - (n_spec - half);
        v = poo[src];
    } else {
        v = make_cuDoubleComplex(0.0, 0.0);
    }
    v.x *= scale_inv;
    v.y *= scale_inv;
    cuDoubleComplex npb = norm_pre_beta[idx];
    double re = v.x * npb.x - v.y * npb.y;
    double im = v.x * npb.y + v.y * npb.x;
    double w = owin[idx];
    ks_out[idx] = make_cuDoubleComplex(re * w, im * w);
}

extern "C" __global__ void weaknorm_reduce_kernel(
    const double* in,
    double* out,
    int n
) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    double sum = 0.0;
    if (i < n) sum = in[i];
    if (i + blockDim.x < n) {
        sum += in[i + blockDim.x];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
