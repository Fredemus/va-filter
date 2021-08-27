use crate::filter_parameters::FilterParameters;
use crate::utils::AtomicOps;
use std::sync::Arc;
#[allow(dead_code)]
#[derive(PartialEq, Clone, Copy)]
// TODO: Flatten jacobian matrices
enum EstimateSource {
    State,               // use current state
    PreviousVout,        // use z-1 of Vout
    LinearStateEstimate, // use linear estimate of future state
    LinearVoutEstimate,  // use linear estimate of Vout
}
pub trait Filter {
    fn new() -> Self;

    fn run_filter_linear(&mut self, input: f32) -> f32;

    fn run_filter_pivotal(&mut self, input: f32) -> f32;

    fn run_filter_newton(&mut self, input: f32) -> f32;

    fn tick_filter(&mut self, input: f32) -> f32;

    fn update_state(&mut self);
}
pub struct LadderFilter {
    pub params: Arc<FilterParameters>,
    vout: [f32; 4],
    s: [f32; 4],
}
#[allow(dead_code)]
impl LadderFilter {
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32) -> f32 {
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
    // nonlinear ladder filter function with distortion.
    fn run_filter_pivotal(&mut self, input: f32) -> f32 {
        let mut a: [f32; 5] = [1.; 5];
        // let base = [input, self.s[0], self.s[1], self.s[2], self.s[3]];
        let g = self.params.g.get();
        let k = self.params.k_ladder.get();
        let base = [
            input - k * self.s[3],
            // input, // <- old base[0]
            self.s[0],
            self.s[1],
            self.s[2],
            self.s[3],
        ];
        // a[n] is the fixed-pivot approximation for tanh()
        for n in 0..base.len() {
            if base[n] != 0. {
                a[n] = base[n].tanh() / base[n];
            }
        }
        // denominators of solutions of individual stages. Simplifies the math a bit
        let g0 = 1. / (1. + g * a[1]);
        let g1 = 1. / (1. + g * a[2]);
        let g2 = 1. / (1. + g * a[3]);
        let g3 = 1. / (1. + g * a[4]);
        //  these are just factored out of the feedback solution. Makes the math way easier to read
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
            / (f0 * k * a[3] + 1.);
        // since we know the feedback, we can solve the remaining outputs:
        self.vout[0] = g0 * (g * a[1] * (input * a[0] - k * a[3] * self.vout[3]) + self.s[0]);
        self.vout[1] = g1 * (g * a[2] * self.vout[0] + self.s[1]);
        self.vout[2] = g2 * (g * a[3] * self.vout[1] + self.s[2]);

        return self.vout[self.params.slope.get()];
    }
    // linear version without distortion
    fn run_filter_linear(&mut self, input: f32) -> f32 {
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
    fn run_filter_newton(&mut self, input: f32) -> f32 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = self.params.g.get();
        let k = self.params.k_ladder.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32; 4];
        let mut temp: [f32; 4] = [0.; 4];

        let est_type = EstimateSource::LinearVoutEstimate;
        // let est_type = EstimateSource::State;

        // getting initial estimate. Could potentially be done with the fixed_pivot filter
        v_est = [
            self.get_estimate(0, est_type, input),
            self.get_estimate(1, est_type, input),
            self.get_estimate(2, est_type, input),
            self.get_estimate(3, est_type, input),
        ];
        let mut tanh_input = (input - k * v_est[3]).tanh();
        let mut tanh_y1_est = v_est[0].tanh();
        let mut tanh_y2_est = v_est[1].tanh();
        let mut tanh_y3_est = v_est[2].tanh();
        let mut tanh_y4_est = v_est[3].tanh();
        let mut residue = [
            g * (tanh_input - tanh_y1_est) + self.s[0] - v_est[0],
            g * (tanh_y1_est - tanh_y2_est) + self.s[1] - v_est[1],
            g * (tanh_y2_est - tanh_y3_est) + self.s[2] - v_est[2],
            g * (tanh_y3_est - tanh_y4_est) + self.s[3] - v_est[3],
        ];
        // println!("residue: {:?}", residue);
        // println!("vest: {:?}", v_est);
        let max_error = 0.00001;
        let mut n_iterations = 0;
        while (residue[0].abs() > max_error
            || residue[1].abs() > max_error
            || residue[2].abs() > max_error
            || residue[3].abs() > max_error)
            && n_iterations < 9
        {
            // if n_iterations > 10 {
            //     break;
            // panic!("filter doesn't converge");
            // }
            // jacobian matrix
            let mut j: [[f32; 4]; 4] = [[0.; 4]; 4];

            j[1][0] = g * (1. - tanh_y1_est * tanh_y1_est);
            j[0][0] = -j[1][0] - 1.;
            j[0][3] = -g * k * (1. - tanh_input * tanh_input);
            j[2][1] = g * (1. - tanh_y2_est * tanh_y2_est);
            j[1][1] = -j[2][1] - 1.;
            j[3][2] = g * (1. - tanh_y3_est * tanh_y3_est);
            j[2][2] = -j[3][2] - 1.;
            j[3][3] = -g * (1. - tanh_y4_est * tanh_y4_est) - 1.;

            // this one is disgustingly huge, but couldn't find a way to avoid that. Look into inverting matrix
            // maybe try replacing j_m_n with the expressions and simplify in maple?
            temp[0] = (((j[2][2] * residue[3] - j[3][2] * residue[2]) * j[1][1]
                + j[2][1] * j[3][2] * (-j[1][0] * v_est[0] + residue[1]))
                * j[0][3]
                + j[1][1] * j[2][2] * j[3][3] * (j[0][0] * v_est[0] - residue[0]))
                / (j[0][0] * j[1][1] * j[2][2] * j[3][3] - j[0][3] * j[1][0] * j[2][1] * j[3][2]);

            temp[1] = (j[1][0] * v_est[0] - j[1][0] * temp[0] + j[1][1] * v_est[1] - residue[1])
                / (j[1][1]);
            temp[2] = (j[2][1] * v_est[1] - j[2][1] * temp[1] + j[2][2] * v_est[2] - residue[2])
                / (j[2][2]);
            temp[3] = (j[3][2] * v_est[2] - j[3][2] * temp[2] + j[3][3] * v_est[3] - residue[3])
                / (j[3][3]);

            v_est = temp;
            tanh_input = (input - k * v_est[3]).tanh();
            tanh_y1_est = v_est[0].tanh();
            tanh_y2_est = v_est[1].tanh();
            tanh_y3_est = v_est[2].tanh();
            tanh_y4_est = v_est[3].tanh();

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
    pub fn tick_newton(&mut self, input: f32) -> f32 {
        // perform filter process
        let out = self.run_filter_newton(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out * (1. + self.params.k_ladder.get())
    }
    // performs a complete filter process (newton-raphson method)
    pub fn tick_pivotal(&mut self, input: f32) -> f32 {
        // perform filter process
        let out = self.run_filter_pivotal(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
}

pub struct SVF {
    pub params: Arc<FilterParameters>,
    vout: [f32; 2],
    s: [f32; 2],
}
#[allow(dead_code)]
impl SVF {
    // the state needs to be updated after each process. Found by trapezoidal integration
    #[inline]
    fn update_state(&mut self) {
        self.s[0] = 2. * self.vout[0] - self.s[0];
        self.s[1] = 2. * self.vout[1] - self.s[1];
    }
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32) -> f32 {
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

    // performs a complete filter process (fixed-pivot method)
    fn tick_pivotal(&mut self, input: f32) -> f32 {
        // perform filter process
        let out = self.run_svf_pivotal(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    // performs a complete filter process (fixed-pivot method)
    pub fn tick_newton(&mut self, input: f32) -> f32 {
        // perform filter process
        let out = self.run_svf_newton(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    pub fn run_svf_linear(&mut self, input: f32) -> f32 {
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
    pub fn run_svf_pivotal(&mut self, input: f32) -> f32 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = self.params.g.get();
        // let k = self.params.res.get();
        let k = self.params.zeta.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut a = [1.; 3];
        let est_type = EstimateSource::State;
        // first getting fixed-pivot approximation for the feedback line, since it's necessary for computing a[0]:
        let est_source_a2 = self.get_estimate(0, est_type, input);
        // employing fixed-pivot method
        if est_source_a2 != 0. {
            // v_t and i_s are constants to control the diode clipper's character
            // just earballed em to be honest. Hard to figure out what they should be
            // without knowing the circuit's operating voltage and temperature
            let v_t = 4.;
            let i_s = 4.;
            // a2 is clipped with the inverse of the diode anti-saturator
            a[2] = (v_t * (est_source_a2 / i_s).asinh()) / est_source_a2;
        }
        let est_source_rest = [
            (input
                - (est_source_a2 * a[2] + (k - 1.) * est_source_a2)
                - self.get_estimate(1, est_type, input)),
            self.get_estimate(0, est_type, input),
        ];
        for n in 0..est_source_rest.len() {
            if est_source_rest[n] != 0. {
                a[n] = est_source_rest[n].tanh() / est_source_rest[n];
            } else {
            }
        }
        // ---------- calculations ----------
        // factored out of the equation
        let g1 = 1. / (g * a[0]);
        let g2 = 1. / (a[0] * a[2] * g * g1 * k - a[0] * a[2] * g * g1 + a[2] * g1 + 1.);
        let g3 = 1. / (1. + g.powi(2) * a[0] * a[1] * g1 * g2 * a[2]);
        // solving equations for output voltages at v1 and v2
        let u = (g * a[0] * input - g * a[0] * self.s[1] + self.s[0]) * g1 * g2 * g3;
        self.vout[0] = u.asinh();
        self.vout[1] = g * a[1] * self.vout[0] + self.s[1];
        // here, the output is chosen to give the specified type of filter
        self.get_output(input, k)
    }
    // trying to avoid having to invert the matrix
    pub fn run_svf_newton(&mut self, input: f32) -> f32 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = self.params.g.get();
        // potentially useful knowledge: filter starts self-oscillating at about k = -0.001
        let k = self.params.zeta.get();
        // let k = self.params.res.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut v_est: [f32; 2];
        let est_type = EstimateSource::State;
        // let est_type = EstimateSource::State;

        // getting initial estimate. Could potentially be done with the fixed_pivot filter
        v_est = [
            self.get_estimate(0, est_type, input),
            self.get_estimate(1, est_type, input),
        ];
        let mut sinh_v_est0 = v_est[0].sinh();
        let mut cosh_v_est0 = v_est[0].cosh();
        let mut tanh_v_est0 = sinh_v_est0 / cosh_v_est0;
        let mut fb_line = (input - ((k - 1.) * v_est[0] + sinh_v_est0) - v_est[1]).tanh();
        // using fixed_pivot as estimate
        // self.run_svf_pivotal(input);
        // v_est = [self.vout[0], self.vout[1]];
        let mut residue = self.run_helper_svf(g, tanh_v_est0, fb_line, v_est);

        let max_error = 0.00001;
        let mut n_iterations = 0;
        while residue[0].abs() > max_error || residue[1].abs() > max_error {
            // terminate if error doesn't improve after 10 iterations
            if n_iterations > 9 {
                // panic!("infinite loop mayhaps?");
                break;
            }
            // TODO: not sure why this can't start out as uninitialized
            let mut jacobian: [[f32; 2]; 2] = [[-1.; 2]; 2];
            
            let new_bigboy = 1. - fb_line * fb_line;
            // TODO: Very likely, the division by tanh_v_est0 causes j[0][0] to go to inf when tanh_v_est0 is 0
            // maybe just bite the bullet and calc cosh_vest0?
            // tanh = sinh / cosh should be safe (overflow problems?), and could be used to get tanh_v_est0
            jacobian[0][0] = -g * new_bigboy * (k - 1. + cosh_v_est0) - 1.;
            jacobian[0][1] = -g * new_bigboy;
            jacobian[1][0] = g * (1. - tanh_v_est0 * tanh_v_est0);

            v_est[0] = (jacobian[0][1] * jacobian[1][0] * v_est[0] + jacobian[0][0] * v_est[0]
                - jacobian[0][1] * residue[1]
                - residue[0])
                / (jacobian[0][1] * jacobian[1][0] + jacobian[0][0]);
            v_est[1] = (jacobian[0][1] * jacobian[1][0] * v_est[1]
                + jacobian[0][0] * residue[1]
                + jacobian[0][0] * v_est[1]
                - jacobian[1][0] * residue[0])
                / (jacobian[0][1] * jacobian[1][0] + jacobian[0][0]);
            sinh_v_est0 = v_est[0].sinh();
            cosh_v_est0 = v_est[0].cosh();
            tanh_v_est0 = sinh_v_est0 / cosh_v_est0;
            fb_line = (input - ((k - 1.) * v_est[0] + sinh_v_est0) - v_est[1]).tanh();
            // recompute filter
            residue = self.run_helper_svf(g, tanh_v_est0, fb_line, v_est);
            n_iterations += 1;
        }
        // when newton's method is done, we have some good estimates for vout
        self.vout = v_est;
        // here, the output is chosen to give the specified type of filter
        self.get_output(input, k)
    }

    /// helper function for newton's method
    #[inline]
    pub fn run_helper_svf(
        &mut self,
        g: f32,
        tanh_v_est0: f32,
        fb_line: f32,
        v_est: [f32; 2],
    ) -> [f32; 2] {
        let residue: [f32; 2] = [
            g * fb_line + self.s[0] - v_est[0],
            g * tanh_v_est0 + self.s[1] - v_est[1],
        ];
        // residue[0] = g * fb_line + self.s[0] - v_est[0];
        // residue[1] = g * tanh_v_est0 + self.s[1] - v_est[1];
        residue
    }
    /// here the output is chosen to give the specified type of filter
    #[inline(always)]
    fn get_output(&self, input: f32, k: f32) -> f32 {
        match self.params.mode.get() {
            0 => self.vout[1],                            // lowpass
            1 => input - k * self.vout[0] - self.vout[1], // highpass
            2 => self.vout[0],                            // bandpass
            3 => input - k * self.vout[0],                // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            4 => input - 2. * self.vout[1] - k * self.vout[0], // peak
            _ => k * self.vout[0],                             // bandpass (normalized peak gain)
        }
    }
}
impl Default for SVF {
    fn default() -> Self {
        Self {
            params: Arc::new(FilterParameters::default()),
            vout: [0.; 2],
            s: [0.; 2],
        }
    }
}
impl Default for LadderFilter {
    fn default() -> Self {
        Self {
            params: Arc::new(FilterParameters::default()),
            vout: [0.; 4],
            s: [0.; 4],
        }
    }
}
#[test]
fn save_filter_impulse() {
    let mut plugin = SVF::default();

    // setting up hound for creating .wav files
    use hound;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(format!("testing/newton_impulse.wav"), spec).unwrap();
    // plugin.params.set_parameter(0, 0.4);
    let len = 100000;
    let mut input_sample = 1.;
    // saving samples to wav file
    for _i in 0..len {
        // let output_sample = plugin.tick_pivotal(input_sample);
        let output_sample = plugin.tick_newton(input_sample);
        // println!("out: {}", plugin.vout[0]);
        writer
            // .write_sample(plugin.tick_newton(input_sample))
            .write_sample(output_sample)
            .unwrap();
        input_sample = 0.0;
    }
}
#[test]
fn newton_svf_test() {
    let mut plugin = SVF::default();

    println!("g: {}", plugin.params.g.get());

    let len = 1;
    let mut input_sample = 0.;
    // saving samples to wav file
    for _i in 0..len {
        plugin.tick_newton(input_sample);

        input_sample = 0.;
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
        println!("{}", plugin.tick_newton(input_sample));
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
        let _sample = plugin.tick_newton(amplitude * (t * 440.0 * 2.0 * 3.14159265).sin());
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
        let _sample = plugin.tick_newton(rng.gen_range(-amplitude..amplitude));
        // let amplitude = i16::MAX as f32;
        // writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    // for _i in 0..len {
    //     plugin.tick_newton(input_sample);

    //     input_sample = 0.;
    // }
}
