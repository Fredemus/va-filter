use crate::filter_parameters::FilterParameters;
use crate::utils::AtomicOps;
use std::sync::Arc;
use packed_simd::f32x4;

/// cheap tanh, potentially useful for optimization. 
// from a quick look it looks extremely good, max error of ~0.0002 or .02%
// the error of 1 - tanh_levien^2 as the derivative is about .06%, so maybe this could easily be substituted?
#[inline]
pub fn tanh_levien(x: f32x4) -> f32x4 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    
    let a = x
        + (0.16489087 * x3)
        + (0.00985468 * x5);
    
    a / (1.0 + (a * a)).sqrt()
}
// from cursory benchmarking, this is as fast as the standard library cosh
#[inline]
fn simd_cosh(x: f32x4) -> f32x4 {
    let e = f32x4::splat(std::f32::consts::E);
    (e.powf(x) + e.powf(-x))/2.
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
    pub params: Arc<FilterParameters>,
    // vout: [f32; 4],
    // s: [f32; 4],

    // s0: f32x4,
    // s1: f32x4,
    // s2: f32x4,
    // s3: f32x4,

    // vout0: f32x4,
    // vout1: f32x4,
    // vout2: f32x4,
    // vout3: f32x4,
    vout: [f32x4; 4],
    s: [f32x4; 4],

}
#[allow(dead_code)]
impl LadderFilter {
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
            EstimateSource::LinearStateEstimate => 2. * self.vout[n] - self.s[n],
            EstimateSource::LinearVoutEstimate => self.vout[n],
        }
    }
    #[inline(always)]
    fn update_state(&mut self) {
        self.s[0] = 2. * self.vout[0] - self.s[0]; 
        self.s[1] = 2. * self.vout[1] - self.s[1]; 
        self.s[2] = 2. * self.vout[2] - self.s[2]; 
        self.s[3] = 2. * self.vout[3] - self.s[3]; 
    }
    // linear version without distortion
    fn run_filter_linear(&mut self, input: f32x4) -> f32x4 {
        // denominators of solutions of individual stages. Simplifies the math a bit
        let g = self.params.g.get();
        let k = self.params.k_ladder.get();
        let g0 = 1. / (1. + g);
        let g1 = g * g0 * g0;
        let g2 = g * g1 * g0;
        let g3 = self.params.g.get() * g2 * g0;
        // outputs a 24db filter
        self.vout[3] =
            (g3 * g * input + g0 * self.s[3] + g1 * self.s[2] + g2 * self.s[1] + g3 * self.s[0])
                / (g3 * g * k + 1.);
        // since we know the feedback, we can solve the remaining outputs:
        self.vout[0] = g0 * (g * (input - k * self.vout[3]) + self.s[0]);
        self.vout[1] = g0 * (g * self.vout[0] + self.s[1]);
        self.vout[2] = g0 * (g * self.vout[1] + self.s[2]);
        return self.vout[self.params.slope.get()];
    }
    fn run_filter_newton(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = self.params.g.get();
        let k = self.params.k_ladder.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32x4; 4];
        let mut temp: [f32x4; 4] = [f32x4::splat(0.); 4];

        let est_type = EstimateSource::LinearVoutEstimate;
        // let est_type = EstimateSource::State;

        // getting initial estimate. Could potentially be done with the fixed_pivot filter
        v_est = [
            self.get_estimate(0, est_type, input),
            self.get_estimate(1, est_type, input),
            self.get_estimate(2, est_type, input),
            self.get_estimate(3, est_type, input),
        ];
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
        if residue[0] == max_error {

        }
        // f32x4.lt(max_error) returns a mask. 
        while (residue[0].abs().gt(max_error).any()
            || residue[1].abs().gt(max_error).any()
            || residue[2].abs().gt(max_error).any()
            || residue[3].abs().gt(max_error).any())
            && n_iterations < 9
        {
            // if n_iterations > 10 {
            //     break;
            // panic!("filter doesn't converge");
            // }

            // jacobian matrix
            let j10 = g * (1. - tanh_y1_est * tanh_y1_est);
            let j00 = - j10 - 1.;
            let j03 = -g * k * (1. - tanh_input * tanh_input);
            let j21 = g * (1. - tanh_y2_est * tanh_y2_est);
            let j11 = - j21 - 1.;
            let j32 = g * (1. - tanh_y3_est * tanh_y3_est);
            let j22 = - j32 - 1.;
            let j33 = -g * (1. - tanh_y4_est * tanh_y4_est) - 1.;

            // this one is disgustingly huge, but couldn't find a way to avoid that. Look into inverting matrix
            // maybe try replacing j_m_n with the expressions and simplify in maple? <- didn't help
            temp[0] = (((j22 * residue[3] - j32 * residue[2]) * j11
                + j21 * j32 * (-j10 * v_est[0] + residue[1]))
                * j03
                + j11 * j22 * j33 * (j00 * v_est[0] - residue[0]))
                / (j00 * j11 * j22 * j33 - j03 * j10 * j21 * j32);

            temp[1] = (j10 * v_est[0] - j10 * temp[0] + j11 * v_est[1] - residue[1])
                / (j11);
            temp[2] = (j21 * v_est[1] - j21 * temp[1] + j22 * v_est[2] - residue[2])
                / (j22);
            temp[3] = (j32 * v_est[2] - j32 * temp[2] + j33 * v_est[3] - residue[3])
                / (j33);

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
            n_iterations += 1;
        }
        // println!("n iterations: {}", n_iterations);
        self.vout = v_est;
        return self.vout[self.params.slope.get()];
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_newton(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_filter_newton(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out * (1. + self.params.k_ladder.get())
    }
    // performs a complete filter process (newton-raphson method)
    
}

pub struct SVF {
    pub params: Arc<FilterParameters>,
    vout: [f32x4; 2],
    s: [f32x4; 2],
}
#[allow(dead_code)]
impl SVF {
    // the state needs to be updated after each process. Found by trapezoidal integration
    #[inline]
    fn update_state(&mut self) {
        self.s[0] = 2. * self.vout[0] - self.s[0];
        self.s[1] = 2. * self.vout[1] - self.s[1];
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
            EstimateSource::LinearStateEstimate => 2. * self.vout[n] - self.s[n],
            EstimateSource::LinearVoutEstimate => self.vout[n],
        }
    }

    // performs a complete filter process (newton-raphson method)
    pub fn tick_newton(&mut self, input: f32x4) -> f32x4 {
        // perform filter process
        let out = self.run_svf_newton(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    pub fn run_svf_linear(&mut self, input: f32x4) -> f32x4 {
        let g = self.params.g.get();
        // declaring some constants that simplifies the math a bit
        // let k = self.params.res.get();
        let k = self.params.zeta.get();
        let g1 = 1. / (1. + g * (g + k));
        let g2 = g * g1;
        // let g3 = g * g2;
        // outputs the correct output voltages
        self.vout[0] = g1 * self.s[0] + g2 * (input - self.s[1]);
        // self.vout[1] = (input - self.s[1]) * g3 + self.s[0] * g2 + self.s[1]; <- meant for parallel processing
        self.vout[1] = self.s[1] + g * self.vout[0];
        self.get_output(input, k)
    }
    // trying to avoid having to invert the matrix
    pub fn run_svf_newton(&mut self, input: f32x4) -> f32x4 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = self.params.g.get();
        // potentially useful knowledge: filter starts self-oscillating at about k = -0.001
        let k = self.params.zeta.get();
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
        let mut fb_line = tanh_levien(input - ((k - 1.) * v_est[0] + sinh_v_est0) - v_est[1]);
        // using fixed_pivot as estimate
        // self.run_svf_pivotal(input);
        // v_est = [self.vout[0], self.vout[1]];
        let mut residue = [
            g * fb_line + self.s[0] - v_est[0],
            g * tanh_v_est0 + self.s[1] - v_est[1],
        ];

        let max_error = f32x4::splat(0.00001);
        let mut n_iterations = 0;
        while (residue[0].abs().gt(max_error).any() || residue[0].abs().gt(max_error).any()) && n_iterations < 9 {
            // terminate if error doesn't improve after 10 iterations
            if n_iterations > 9 {
                // panic!("infinite loop mayhaps?");
                break;
            }
            
            let new_bigboy = 1. - fb_line * fb_line;
            // TODO: Very likely, the division by tanh_v_est0 causes j[0][0] to go to inf when tanh_v_est0 is 0
            // maybe just bite the bullet and calc cosh_vest0?
            // tanh = sinh / cosh should be safe (overflow problems?), and could be used to get tanh_v_est0
            // jacobian matrix
            let j00 = -g * new_bigboy * (k - 1. + cosh_v_est0) - 1.;
            let j01 = -g * new_bigboy;
            let j10 = g * (1. - tanh_v_est0 * tanh_v_est0);

            v_est[0] = (j01 * j10 * v_est[0] + j00 * v_est[0]
                - j01 * residue[1]
                - residue[0])
                / (j01 * j10 + j00);
            v_est[1] = (j01 * j10 * v_est[1]
                + j00 * residue[1]
                + j00 * v_est[1]
                - j10 * residue[0])
                / (j01 * j10 + j00);
            cosh_v_est0 = simd_cosh(v_est[0]);
            tanh_v_est0 = tanh_levien(v_est[0]);
            sinh_v_est0 = tanh_v_est0 * cosh_v_est0;
            fb_line = tanh_levien(input - ((k - 1.) * v_est[0] + sinh_v_est0) - v_est[1]);
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
    fn get_output(&self, input: f32x4, k: f32) -> f32x4 {
        match self.params.mode.get() {
            0 => self.vout[1],                            // lowpass
            1 => input - k * self.vout[0] - self.vout[1], // highpass
            2 => self.vout[0],                            // bandpass
            3 => input - k * self.vout[0],                // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            4 => k * self.vout[0],                             // bandpass (normalized peak gain)
            _ => input - 2. * self.vout[1] - k * self.vout[0], // peak
        }
    }
}
impl Default for SVF {
    fn default() -> Self {
        Self {
            params: Arc::new(FilterParameters::default()),
            vout: [f32x4::splat(0.);2],
            s: [f32x4::splat(0.);2],
        }
    }
}
impl Default for LadderFilter {
    fn default() -> Self {
        Self {
            params: Arc::new(FilterParameters::default()),
            vout: [f32x4::splat(0.);4],
            s: [f32x4::splat(0.); 4],
            // vout: [0.; 4],
            // s: [0.; 4],
        }
    }
}
// #[test]
// fn save_filter_impulse() {
//     let mut plugin = SVF::default();

//     // setting up hound for creating .wav files
//     use hound;
//     let spec = hound::WavSpec {
//         channels: 1,
//         sample_rate: 44100,
//         bits_per_sample: 32,
//         sample_format: hound::SampleFormat::Float,
//     };
//     let writer = hound::WavWriter::create(format!("testing/newton_impulse.wav"), spec).unwrap();
//     // plugin.params.set_parameter(0, 0.4);
//     let len = 100000;
//     let mut input_sample = f32x4::splat(1.);
//     // saving samples to wav file
//     for _i in 0..len {
//         // let output_sample = plugin.tick_pivotal(input_sample);
//         let output_sample = plugin.tick_newton(input_sample);
//         // println!("out: {}", plugin.vout[0]);
//         writer
//             .write_sample(output_sample)
//             .unwrap();
//         input_sample = f32x4::splat(0.0);
//     }
// }
#[test]
fn newton_svf_test() {
    let mut plugin = SVF::default();

    println!("g: {}", plugin.params.g.get());

    let len = 1;
    let mut input_sample = f32x4::splat(0.);
    // saving samples to wav file
    for _i in 0..len {
        plugin.tick_newton(input_sample);

        input_sample = f32x4::splat(0.);
    }
}
#[test]
fn newton_ladder_test() {
    let mut plugin = LadderFilter::default();

    println!("g: {}", plugin.params.g.get());
    plugin.params.set_resonances();
    // let k = plugin.params.res.get();
    println!("reso: {}", plugin.params.k_ladder.get());
    plugin.params.mode.set_normalized(0.6);
    println!("mode: {}", plugin.params.mode.get());
    let len = 100;
    let mut input_sample = 1.;
    // saving samples to wav file
    for _i in 0..len {
        println!("{:?}", plugin.tick_newton(f32x4::new(input_sample, 0.0, 0.0, 0.0)));
        input_sample = 0.;
    }
}
#[test]
fn newton_test_sine() {
    let mut plugin = SVF::default();

    // println!("g: {}", plugin.params.g.get());
    use crate::vst::plugin::PluginParameters;
    plugin.params.set_parameter(0, 1.);
    // plugin.params.set_parameter(1, 1.);
    // println!("g: {}", plugin.params.g.get());

    let len = 1000;
    let amplitude = 25.;
    // saving samples to wav file
    for t in (0..len).map(|x| x as f32 / 48000.) {
        let _sample = plugin.tick_newton(f32x4::splat(amplitude * (t * 440.0 * 2.0 * 3.14159265).sin()));
        // println!("got here");
        // let amplitude = i16::MAX as f32;
        // writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    // for _i in 0..len {
    //     plugin.tick_newton(input_sample);

    //     input_sample = 0.;
    // }
}
// this test was mostly used for checking convergence
#[test]
fn newton_test_noise_loud() {
    use rand::Rng;
    let mut plugin = SVF::default();
    let mut rng = rand::thread_rng();
    plugin.params.sample_rate.set(48000.);
    // println!("g: {}", plugin.params.g.get());
    // plugin.params.set_parameter(0, 1.);
    // plugin.params.set_parameter(1, 1.);
    // println!("g: {}", plugin.params.g.get());
    let len = 1000;
    let amplitude = 25.;
    // saving samples to wav file
    for _t in (0..len).map(|x| x as f32 / 48000.) {
        let _sample = plugin.tick_newton(f32x4::splat(rng.gen_range(-amplitude..amplitude)));
        // let amplitude = i16::MAX as f32;
        // writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    // for _i in 0..len {
    //     plugin.tick_newton(input_sample);

    //     input_sample = 0.;
    // }
}
