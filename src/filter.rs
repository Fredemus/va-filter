// use crate::filter_parameters::FilterParameters;
// use crate::filter_params_nih::{FilterParams, SvfMode};
use crate::{filter_params::FilterParams, utils::AtomicOps};
// use packed_simd::f32x4;
use core_simd::*;
use std::sync::Arc;
use std_float::*;

mod solver;
use solver::DKSolver;

pub mod sallen_key;
pub mod svf;

/// cheap tanh to make the filter faster.
// from a quick look it looks extremely good, max error of ~0.0002 or .02%
// the error of 1 - tanh_levien^2 as the derivative is about .06%
#[inline]
pub fn tanh_levien(x: f32x4) -> f32x4 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let a = x + (f32x4::splat(0.16489087) * x3) + (f32x4::splat(0.00985468) * x5);
    // println!("a: {:?}, b: {:?}", a, b);
    a / (f32x4::splat(1.0) + (a * a)).sqrt()
}

#[allow(dead_code)]
#[derive(PartialEq, Clone, Copy)]
enum EstimateSource {
    State,               // use current state
    PreviousVout,        // use z-1 of Vout
    LinearStateEstimate, // use linear estimate of future state
    LinearVoutEstimate,  // use linear estimate of Vout
}
pub struct LadderFilter {
    pub params: Arc<FilterParams>,

    vout: [f32x4; 4],
    pub s: [f32x4; 4],
}
#[allow(dead_code)]
impl LadderFilter {
    pub fn new(params: Arc<FilterParams>) -> Self {
        Self {
            params,
            vout: [f32x4::splat(0.); 4],
            s: [f32x4::splat(0.); 4],
        }
    }
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32x4) -> f32x4 {
        // if we ask for an estimate based on the linear filter, we have to run it
        if estimate == EstimateSource::LinearStateEstimate
            || estimate == EstimateSource::LinearVoutEstimate
        {
            self.run_filter_linear(input);
        }
        match estimate {
            EstimateSource::State => self.s[n],
            EstimateSource::PreviousVout => self.vout[n],
            EstimateSource::LinearStateEstimate => f32x4::splat(2.) * self.vout[n] - self.s[n],
            EstimateSource::LinearVoutEstimate => self.vout[n],
        }
    }
    #[inline(always)]
    fn update_state(&mut self) {
        let two = f32x4::splat(2.);
        self.s[0] = two * self.vout[0] - self.s[0];
        self.s[1] = two * self.vout[1] - self.s[1];
        self.s[2] = two * self.vout[2] - self.s[2];
        self.s[3] = two * self.vout[3] - self.s[3];
    }
    // nonlinear ladder filter function with distortion, solved with Mystran's fixed-pivot method.
    fn run_filter_pivotal(&mut self, input: f32x4) -> f32x4 {
        let mut a: [f32x4; 5] = [f32x4::splat(1.); 5];
        // let base = [input, self.s[0], self.s[1], self.s[2], self.s[3]];
        let g = f32x4::splat(self.params.g.get());
        let k = f32x4::splat(self.params.k_ladder.get());
        let base = [
            input - k * self.s[3],
            self.s[0],
            self.s[1],
            self.s[2],
            self.s[3],
        ];
        // a[n] is the fixed-pivot approximation for tanh()
        for n in 0..base.len() {
            // hopefully this should cook down to the original when not 0,
            // and 1 when 0
            let mask = base[n].simd_ne(f32x4::splat(0.));
            a[n] = tanh_levien(base[n]) / base[n];
            // since the line above can become NaN or other stuff when a value in base[n] is 0,
            // replace values where a[n] is 0.
            a[n] = mask.select(a[n], f32x4::splat(1.));
        }
        // denominators of solutions of individual stages. Simplifies the math a bit
        let one = f32x4::splat(1.);
        let g0 = one / (one + g * a[1]);
        let g1 = one / (one + g * a[2]);
        let g2 = one / (one + g * a[3]);
        let g3 = one / (one + g * a[4]);
        // these are factored out of the feedback solution. Makes the math easier to read
        let f3 = g * a[3] * g3;
        let f2 = g * a[2] * g2 * f3;
        let f1 = g * a[1] * g1 * f2;
        let f0 = g * g0 * f1;
        // outputs a 24db filter
        self.vout[3] = (f0 * input * a[0]
            + f1 * g0 * self.s[0]
            + f2 * g1 * self.s[1]
            + f3 * g2 * self.s[2]
            + g3 * self.s[3])
            / (f0 * k * a[3] + one);
        // since we know the feedback, we can solve the remaining outputs:
        self.vout[0] = g0 * (g * a[1] * (input * a[0] - k * a[3] * self.vout[3]) + self.s[0]);
        self.vout[1] = g1 * (g * a[2] * self.vout[0] + self.s[1]);
        self.vout[2] = g2 * (g * a[3] * self.vout[1] + self.s[2]);

        self.vout[self.params.slope.value() as usize]
    }
    // linear version without distortion
    pub fn run_filter_linear(&mut self, input: f32x4) -> f32x4 {
        // denominators of solutions of individual stages. Simplifies the math a bit
        let g = f32x4::splat(self.params.g.get());
        let k = f32x4::splat(self.params.k_ladder.get());
        let one = f32x4::splat(1.);
        let g0 = one / (one + g);
        let g1 = g * g0 * g0;
        let g2 = g * g1 * g0;
        let g3 = g * g2 * g0;
        // outputs a 24db filter
        self.vout[3] =
            (g3 * g * input + g0 * self.s[3] + g1 * self.s[2] + g2 * self.s[1] + g3 * self.s[0])
                / (g3 * g * k + one);
        // since we know the feedback, we can solve the remaining outputs:
        self.vout[0] = g0 * (g * (input - k * self.vout[3]) + self.s[0]);
        self.vout[1] = g0 * (g * self.vout[0] + self.s[1]);
        self.vout[2] = g0 * (g * self.vout[1] + self.s[2]);
        self.vout[self.params.slope.value() as usize]
    }
    pub fn run_filter_newton(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = f32x4::splat(self.params.g.get());
        let k = f32x4::splat(self.params.k_ladder.get());
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32x4; 4];
        let mut temp: [f32x4; 4] = [f32x4::splat(0.); 4];

        // use state as estimate
        v_est = [self.s[0], self.s[1], self.s[2], self.s[3]];

        let mut tanh_input = tanh_levien(input - k * v_est[3]);
        let mut tanh_y1_est = tanh_levien(v_est[0]);
        let mut tanh_y2_est = tanh_levien(v_est[1]);
        let mut tanh_y3_est = tanh_levien(v_est[2]);
        let mut tanh_y4_est = tanh_levien(v_est[3]);
        let mut residue = [
            g * (tanh_input - tanh_y1_est) + self.s[0] - v_est[0],
            g * (tanh_y1_est - tanh_y2_est) + self.s[1] - v_est[1],
            g * (tanh_y2_est - tanh_y3_est) + self.s[2] - v_est[2],
            g * (tanh_y3_est - tanh_y4_est) + self.s[3] - v_est[3],
        ];
        // let max_error = 0.00001;
        let max_error = f32x4::splat(0.00001);

        // f32x4.lt(max_error) returns a mask.
        while residue[0].abs().simd_gt(max_error).any()
            || residue[1].abs().simd_gt(max_error).any()
            || residue[2].abs().simd_gt(max_error).any()
            || residue[3].abs().simd_gt(max_error).any()
        // && n_iterations < 9
        {
            let one = f32x4::splat(1.);
            // jacobian matrix
            let j10 = g * (one - tanh_y1_est * tanh_y1_est);
            let j00 = -j10 - one;
            let j03 = -g * k * (one - tanh_input * tanh_input);
            let j21 = g * (one - tanh_y2_est * tanh_y2_est);
            let j11 = -j21 - one;
            let j32 = g * (one - tanh_y3_est * tanh_y3_est);
            let j22 = -j32 - one;
            let j33 = -g * (one - tanh_y4_est * tanh_y4_est) - one;

            temp[0] = (((j22 * residue[3] - j32 * residue[2]) * j11
                + j21 * j32 * (-j10 * v_est[0] + residue[1]))
                * j03
                + j11 * j22 * j33 * (j00 * v_est[0] - residue[0]))
                / (j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32);

            temp[1] = (j10 * v_est[0] - j10 * temp[0] + j11 * v_est[1] - residue[1]) / (j11);
            temp[2] = (j21 * v_est[1] - j21 * temp[1] + j22 * v_est[2] - residue[2]) / (j22);
            temp[3] = (j32 * v_est[2] - j32 * temp[2] + j33 * v_est[3] - residue[3]) / (j33);

            v_est = temp;
            tanh_input = tanh_levien(input - k * v_est[3]);
            tanh_y1_est = tanh_levien(v_est[0]);
            tanh_y2_est = tanh_levien(v_est[1]);
            tanh_y3_est = tanh_levien(v_est[2]);
            tanh_y4_est = tanh_levien(v_est[3]);

            residue = [
                g * (tanh_input - tanh_y1_est) + self.s[0] - v_est[0],
                g * (tanh_y1_est - tanh_y2_est) + self.s[1] - v_est[1],
                g * (tanh_y2_est - tanh_y3_est) + self.s[2] - v_est[2],
                g * (tanh_y3_est - tanh_y4_est) + self.s[3] - v_est[3],
            ];
            // n_iterations += 1;
        }
        self.vout = v_est;
        self.vout[self.params.slope.value() as usize]
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_newton(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_filter_newton(input * f32x4::splat(self.params.drive.value()));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out * f32x4::splat((1. + self.params.k_ladder.get()) / (self.params.drive.value() * 0.5))
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_pivotal(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_filter_pivotal(input * f32x4::splat(self.params.drive.value()));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_linear(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        // let out = self.run_filter_linear(input * f32x4::splat(self.params.drive.value));
        let out = self.run_filter_linear(input);
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
}
