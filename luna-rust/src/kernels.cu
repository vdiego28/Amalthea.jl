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
    int* err_code
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    double abs_e = fabs(fields[idx]);
    if (abs_e < e_min) {
        rates[idx] = 0.0;
        return;
    }
    
    if (abs_e > e_max) {
        rates[idx] = -1.0;
        atomicExch(err_code, 1);
        return;
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
