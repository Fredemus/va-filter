// use crate::filter_parameters::FilterParameters;
// use crate::filter_params_nih::{FilterParams, SvfMode};
use crate::{utils::AtomicOps, filter_params_nih::{FilterParams, SvfMode}};
// use packed_simd::f32x4;
use core_simd::f32x4;
use std::sync::Arc;
use std_float::*;

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
// TODO: scalar exp and ln should really be replaced when portable_simd adds them
#[inline]
fn simd_cosh(x: f32x4) -> f32x4 {
    (f32x4::from_array(x.to_array().map(f32::exp))
        + f32x4::from_array((-x).to_array().map(f32::exp)))
        / f32x4::splat(2.)
}
#[inline]
fn simd_asinh(x: f32x4) -> f32x4 {
    let mask = x.lanes_gt(f32x4::splat(0.));
    // let val = (x.abs() + ((x * x) + f32x4::splat(1.0)).sqrt()).ln();
    let mut val = x.abs() + ((x * x) + f32x4::splat(1.0)).sqrt();
    val = f32x4::from_array(val.to_array().map(f32::ln));
    // this be equivalent to copysign, preserving the sign of the input. Untested
    mask.select(val, -val)
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
    s: [f32x4; 4],
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
            let mask = base[n].lanes_ne(f32x4::splat(0.));
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

        self.vout[self.params.slope.value() as usize ]
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
        // println!("residue: {:?}", residue);
        // println!("vest: {:?}", v_est);
        // let max_error = 0.00001;
        let max_error = f32x4::splat(0.00001);
        let mut n_iterations = 0;

        // f32x4.lt(max_error) returns a mask.
        while residue[0].abs().lanes_gt(max_error).any()
            || residue[1].abs().lanes_gt(max_error).any()
            || residue[2].abs().lanes_gt(max_error).any()
            || residue[3].abs().lanes_gt(max_error).any()
        // && n_iterations < 9
        {
            n_iterations += 1;
            if n_iterations > 9 {
                println!("TOO MANY ITS");
            }
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

            // this one is disgustingly huge, but couldn't find a way to avoid that. Look into inverting matrix
            // maybe try replacing j_m_n with the expressions and simplify in maple? <- didn't help
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
        let out = self.run_filter_newton(input * f32x4::splat(self.params.drive.value + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out * f32x4::splat((1. + self.params.k_ladder.get()) / (self.params.drive.value * 0.5 + 1.))
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_pivotal(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_filter_pivotal(input * f32x4::splat(self.params.drive.value + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_linear(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        // let out = self.run_filter_linear(input * f32x4::splat(self.params.drive.value + 1.));
        let out = self.run_filter_linear(input);
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
}
// this is a 2-pole filter with resonance, which is why there's 2 states and vouts
pub struct SVF {
    pub params: Arc<FilterParams>,
    vout: [f32x4; 2],
    s: [f32x4; 2],
}
#[allow(dead_code)]
impl SVF {
    pub fn new(params: Arc<FilterParams>) -> Self {
        Self {
            params,
            vout: [f32x4::splat(0.); 2],
            s: [f32x4::splat(0.); 2],
        }
    }
    // the state needs to be updated after each process. Found by trapezoidal integration
    #[inline]
    fn update_state(&mut self) {
        let two = f32x4::splat(2.);
        self.s[0] = two * self.vout[0] - self.s[0];
        self.s[1] = two * self.vout[1] - self.s[1];
    }
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32x4) -> f32x4 {
        // if we ask for an estimate based on the linear filter, we have to run it
        if estimate == EstimateSource::LinearStateEstimate
            || estimate == EstimateSource::LinearVoutEstimate
        {
            self.run_svf_linear(input);
        }
        match estimate {
            EstimateSource::State => self.s[n],
            EstimateSource::PreviousVout => self.vout[n],
            EstimateSource::LinearStateEstimate => f32x4::splat(2.) * self.vout[n] - self.s[n],
            EstimateSource::LinearVoutEstimate => self.vout[n],
        }
    }
    // performs a complete filter process (fixed-pivot method)
    fn tick_pivotal(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_svf_pivotal(input * f32x4::splat(self.params.drive.value + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    // performs a complete filter process (Newton's method)
    pub fn tick_newton(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let mut out = self.run_svf_newton(input * f32x4::splat(self.params.drive.value + 1.));
        // staturating the output and adding some gain compensation for drive
        // should be similar to the EDP Wasp filter
        // TODO: check if tanh distortion feels too strong. If so, implement a simd_asinh or smth
        out = tanh_levien(out * f32x4::splat(0.5))
            * f32x4::splat(2. / (self.params.drive.value * 0.5 + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();

        out
    }
    pub fn run_svf_linear(&mut self, input: f32x4) -> f32x4 {
        let g = f32x4::splat(self.params.g.get());
        // let k = self.params.res.get();
        let k = f32x4::splat(self.params.zeta.get());
        let one = f32x4::splat(1.);
        // declaring some constants that simplifies the math a bit
        let g1 = one / (one + g * (g + k));
        let g2 = g * g1;
        // let g3 = g * g2;
        // find the correct output voltages
        self.vout[0] = g1 * self.s[0] + g2 * (input - self.s[1]);
        // self.vout[1] = (input - self.s[1]) * g3 + self.s[0] * g2 + self.s[1]; <- meant for parallel processing
        self.vout[1] = self.s[1] + g * self.vout[0];
        // get output according to the current filter_mode
        self.get_output(input, k)
    }
    pub fn run_svf_pivotal(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = f32x4::splat(self.params.g.get());
        // let k = self.params.res.get();
        let k = f32x4::splat(self.params.zeta.get());
        let one = f32x4::splat(1.);

        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut a = [f32x4::splat(1.); 3];
        let est_type = EstimateSource::State;
        // first getting fixed-pivot approximation for the feedback line, since it's necessary for computing a[0]:
        let est_source_a2 = self.get_estimate(0, est_type, input);
        // employing fixed-pivot method
        // a[2] first, since it involves the antisaturator
        let mask1 = est_source_a2.lanes_ne(f32x4::splat(0.));
        // a2 is clipped with the inverse of the diode anti-saturator
        // a[2] = (v_t * simd_asinh(est_source_a2 / i_s)) / est_source_a2;
        a[2] = simd_asinh(est_source_a2) / est_source_a2;
        // using the mask to remove any elements that have had division by 0
        a[2] = mask1.select(a[2], f32x4::splat(1.));
        let est_source_rest = [
            (input
                - (est_source_a2 * a[2] + (k - one) * est_source_a2)
                - self.get_estimate(1, est_type, input)),
            self.get_estimate(0, est_type, input),
        ];
        for n in 0..est_source_rest.len() {
            let mask = est_source_rest[n].lanes_ne(f32x4::splat(0.));
            a[n] = tanh_levien(est_source_rest[n]) / est_source_rest[n];
            // since the line above can become NaN or other stuff when a value in base[n] is 0,
            // replace values where a[n] is 0.
            a[n] = mask.select(a[n], f32x4::splat(1.));
        }
        // ---------- calculations ----------
        // factored out of the equation
        let g1 = one / (g * a[0]);
        let g2 = one / (a[0] * a[2] * g * g1 * k - a[0] * a[2] * g * g1 + a[2] * g1 + one);
        let g3 = one / (one + g * g * a[0] * a[1] * g1 * g2 * a[2]);
        // solving equations for output voltages at v1 and v2
        let u = (g * a[0] * input - g * a[0] * self.s[1] + self.s[0]) * g1 * g2 * g3;
        self.vout[0] = simd_asinh(u);
        self.vout[1] = g * a[1] * self.vout[0] + self.s[1];
        // here, the output is chosen to give the specified type of filter
        self.get_output(input, k)
    }
    // trying to avoid having to invert the matrix
    pub fn run_svf_newton(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = f32x4::splat(self.params.g.get());
        // potentially useful knowledge: filter starts self-oscillating at about k = -0.001
        let k = f32x4::splat(self.params.zeta.get());
        let one = f32x4::splat(1.);
        // let k = self.params.res.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32x4; 2];
        let est_type = EstimateSource::State;
        // let est_type = EstimateSource::State;

        // getting initial estimate. Could potentially be done with the fixed_pivot filter
        v_est = [
            self.get_estimate(0, est_type, input),
            self.get_estimate(1, est_type, input),
        ];
        // let mut sinh_v_est0 = v_est[0].sinh();
        // let mut cosh_v_est0 = v_est[0].cosh();
        let mut tanh_v_est0 = tanh_levien(v_est[0]);
        let mut cosh_v_est0 = simd_cosh(v_est[0]);
        let mut sinh_v_est0 = tanh_v_est0 * cosh_v_est0; // from a trig identity
        let mut fb_line = tanh_levien(input - ((k - one) * v_est[0] + sinh_v_est0) - v_est[1]);
        // using fixed_pivot as estimate
        // self.run_svf_pivotal(input);
        // v_est = [self.vout[0], self.vout[1]];
        let mut residue = [
            g * fb_line + self.s[0] - v_est[0],
            g * tanh_v_est0 + self.s[1] - v_est[1],
        ];

        let max_error = f32x4::splat(0.00001);
        let mut n_iterations = 0;
        while (residue[0].abs().lanes_gt(max_error).any()
            || residue[0].abs().lanes_gt(max_error).any())
            && n_iterations < 9
        {
            // terminate if error doesn't improve after 10 iterations
            // if n_iterations > 9 {
            //     // panic!("infinite loop mayhaps?");
            //     break;
            // }
            let new_bigboy = one - fb_line * fb_line;
            // jacobian matrix
            let j00 = -g * new_bigboy * (k - one + cosh_v_est0) - one;
            let j01 = -g * new_bigboy;
            let j10 = g * (one - tanh_v_est0 * tanh_v_est0);

            v_est[0] = (j01 * j10 * v_est[0] + j00 * v_est[0] - j01 * residue[1] - residue[0])
                / (j01 * j10 + j00);
            v_est[1] = (j01 * j10 * v_est[1] + j00 * residue[1] + j00 * v_est[1]
                - j10 * residue[0])
                / (j01 * j10 + j00);
            cosh_v_est0 = simd_cosh(v_est[0]);
            tanh_v_est0 = tanh_levien(v_est[0]);
            sinh_v_est0 = tanh_v_est0 * cosh_v_est0;
            fb_line =
                tanh_levien(input - ((k - f32x4::splat(1.)) * v_est[0] + sinh_v_est0) - v_est[1]);
            // recompute filter
            // residue = self.run_helper_svf(g, tanh_v_est0, fb_line, v_est);
            residue = [
                g * fb_line + self.s[0] - v_est[0],
                g * tanh_v_est0 + self.s[1] - v_est[1],
            ];
            n_iterations += 1;
        }
        // when newton's method is done, we have some good estimates for vout
        self.vout = v_est;
        // here, the output is chosen to give the specified type of filter
        self.get_output(input, k)
    }

    /// here the output is chosen to give the specified type of filter
    #[inline(always)]
    fn get_output(&self, input: f32x4, k: f32x4) -> f32x4 {
        match self.params.mode.value() {
            SvfMode::LP => self.vout[1],                            // lowpass
            SvfMode::HP => input - k * self.vout[0] - self.vout[1], // highpass
            SvfMode::BP1 => self.vout[0],                            // bandpass
            SvfMode::Notch => input - k * self.vout[0],                // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            SvfMode::BP2 => k * self.vout[0], // bandpass (normalized peak gain)
            // _ => input - f32x4::splat(2.) * self.vout[1] - k * self.vout[0], // peak / resonator thingy
        }
    }
}
// impl Default for SVF {
//     fn default() -> Self {
//         Self {
//             params: Arc::new(FilterParams::new()),
//             vout: [f32x4::splat(0.); 2],
//             s: [f32x4::splat(0.); 2],
//         }
//     }
// }
// impl Default for LadderFilter {
//     fn default() -> Self {
//         Self {
//             params: Arc::new(FilterParams::default()),
//             vout: [f32x4::splat(0.); 4],
//             s: [f32x4::splat(0.); 4],
//         }
//     }
// }
