//! This zero-delay feedback filter is based on a state variable filter.
//! It follows the following equations:
//!
//! Since we can't easily solve a nonlinear equation,
//! Mystran's fixed-pivot method is used to approximate the tanh() parts.
//! Quality can be improved a lot by oversampling a bit.
//! Damping feedback is antisaturated, so it doesn't disappear at high gains.

#[macro_use]
extern crate vst;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use vst::buffer::AudioBuffer;
use vst::plugin::{Category, Info, Plugin, PluginParameters};
use vst::util::AtomicFloat;
enum _Mode {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Peak,
}
#[allow(dead_code)]
#[derive(PartialEq, Clone, Copy)]
enum EstimateSource {
    State,               // use current state
    PreviousVout,        // use z-1 of Vout
    LinearStateEstimate, // use linear estimate of future state
    LinearVoutEstimate,  // use linear estimate of Vout
}

// this is a 4-pole filter with resonance, which is why there's 4 states and vouts
#[derive(Clone)]
struct SVF {
    // Store a handle to the plugin's parameter object.
    params: Arc<FilterParameters>,
    // the output of the different filter stages
    vout: [f32; 2],
    // s is the "state" parameter. In an IIR it would be the last value from the filter
    // In this we find it by trapezoidal integration to avoid the unit delay
    s: [f32; 2],
}
struct FilterParameters {
    // the "cutoff" parameter. Determines how heavy filtering is
    cutoff: AtomicFloat,
    g: AtomicFloat,
    // needed to calculate cutoff.
    sample_rate: AtomicFloat,
    // makes a peak at cutoff
    res: AtomicFloat,
    // a drive parameter. Just used to increase the volume, which results in heavier distortion
    drive: AtomicFloat,
    // mode parameter. Chooses the correct output from the svf filter
    mode: AtomicUsize,
}
impl Default for FilterParameters {
    fn default() -> FilterParameters {
        FilterParameters {
            cutoff: AtomicFloat::new(1000.),
            res: AtomicFloat::new(2.),
            drive: AtomicFloat::new(0.),
            sample_rate: AtomicFloat::new(44100.),
            g: AtomicFloat::new(0.07135868),
            mode: AtomicUsize::new(0),
        }
    }
}
// member methods for the struct
#[allow(dead_code)]
impl SVF {
    // the state needs to be updated after each process. Found by trapezoidal integration
    fn update_state(&mut self) {
        self.s[0] = 2. * self.vout[0] - self.s[0];
        self.s[1] = 2. * self.vout[1] - self.s[1];
    }
    fn get_estimate(&mut self, n: usize, estimate: EstimateSource, input: f32) -> f32 {
        // if we ask for an estimate based on the linear filter, we have to run it
        if estimate == EstimateSource::LinearStateEstimate
            || estimate == EstimateSource::LinearVoutEstimate
        {
            self.run_svf_linear(input * (self.params.drive.get() + 1.));
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
        let out = self.run_svf_nonlinear(input * (self.params.drive.get() + 1.));
        // update ic1eq and ic2eq for next sample
        self.update_state();
        out
    }
    pub fn run_svf_linear(&mut self, input: f32) -> f32 {
        let g = self.params.g.get();
        // declaring some constants that simplifies the math a bit
        let k = self.params.res.get();
        let g1 = 1. / (1. + g * (g + k));
        let g2 = g * g1;
        // let g3 = g * g2;
        // outputs the correct output voltages
        self.vout[0] = g1 * self.s[0] + g2 * (input - self.s[1]);
        // self.vout[1] = (input - self.s[1]) * g3 + self.s[0] * g2 + self.s[1]; <- meant for parallel processing
        self.vout[1] = self.s[1] + g * self.vout[0];
        match self.params.mode.load(Ordering::Relaxed) {
            0 => self.vout[1],
            1 => input - k * self.vout[0] - self.vout[1],
            2 => self.vout[0],
            3 => input - k * self.vout[0],
            //3 => input - 2. * k * self.vout[1], // <- allpass
            _ => input - 2. * self.vout[1] - k * self.vout[0],
        }
    }
    pub fn run_svf_nonlinear(&mut self, input: f32) -> f32 {
        // ---------- setup ----------
        // load in g and k from parameters
        let g = self.params.g.get();
        let k = self.params.res.get();
        // a[n] is the fixed-pivot approximation for whatever is being processed nonlinearly
        let mut a = [1.; 3];
        let est_type = EstimateSource::State;
        // first getting fixed-pivot approximation for the feedback line, since it's necessary for computing a[0]:
        let est_source_a2 = self.get_estimate(0, est_type, input);
        // employing fixed-pivot method
        if est_source_a2 != 0. {
            // v_t and i_s are constants to control the diode clipper's character
            let v_t = 4.;
            let i_s = 4.;
            // TODO: tanh() might not be proper here. This is the diode clipper
            a[2] = (v_t * (est_source_a2 / i_s).asinh()) / est_source_a2;
            // a[0] = ((est_source_a0).tanh()) / est_source_a0;
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
        match self.params.mode.load(Ordering::Relaxed) {
            0 => self.vout[1],                            // lowpass
            1 => input - k * self.vout[0] - self.vout[1], // highpass
            2 => self.vout[0],                            // bandpass
            3 => input - k * self.vout[0],                // notch
            //3 => input - 2. * k * self.vout[1], // allpass
            _ => input - 2. * self.vout[1] - k * self.vout[0], // peak
        }
    }
}
impl FilterParameters {
    pub fn set_cutoff(&self, value: f32) {
        // cutoff formula gives us a natural feeling cutoff knob that spends more time in the low frequencies
        // this parameter is for viewing by the user
        self.cutoff.set(20000. * (1.8f32.powf(10. * value - 10.)));
        // bilinear transformation for g gives us a very accurate cutoff.
        // this is the parameter that the filter actually uses
        self.g
            .set((PI * self.cutoff.get() / (self.sample_rate.get())).tan());
    }
    pub fn set_res(&self, value: f32) {
        // res is equivalent to 2 * zeta, or 1/Q-factor.
        // the specific formula is scaled so it feels natural to tweak the parameter
        self.res.set(100. * (2f32.powf(-11. * value)))
    }
    pub fn get_res(&self) -> f32 {
        -0.1311540946 * (0.01 * self.res.get()).ln()
    }
    // returns the value used to set cutoff. for get_parameter function
    pub fn get_cutoff(&self) -> f32 {
        1. + 0.1701297528 * (0.00005 * self.cutoff.get()).ln()
    }
    pub fn set_mode(&self, value: f32) {
        let val: usize = (value * 5.).round() as usize;
        self.mode.store(val, Ordering::Relaxed);
    }
    fn get_mode(&self) -> f32 {
        self.mode.load(Ordering::Relaxed) as f32 / 5.
    }
}
impl PluginParameters for FilterParameters {
    // get_parameter has to return the value used in set_parameter. Used for preset loading and such
    fn get_parameter(&self, index: i32) -> f32 {
        match index {
            0 => self.get_cutoff(),
            1 => self.get_res(),
            2 => self.drive.get() / 5.,
            3 => self.get_mode(),
            _ => 0.0,
        }
    }
    fn set_parameter(&self, index: i32, value: f32) {
        match index {
            0 => self.set_cutoff(value),
            1 => self.set_res(value),
            2 => self.drive.set(value * 16.),
            3 => self.set_mode(value),
            _ => (),
        }
    }
    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            0 => "cutoff".to_string(),
            1 => "resonance".to_string(),
            2 => "drive".to_string(),
            3 => "filter mode".to_string(),
            _ => "".to_string(),
        }
    }
    fn get_parameter_label(&self, index: i32) -> String {
        match index {
            0 => "Hz".to_string(),
            1 => "%".to_string(),
            2 => "dB".to_string(),
            _ => "".to_string(),
        }
    }
    // This is what will display underneath our control.  We can
    // format it into a string that makes sense for the user.
    fn get_parameter_text(&self, index: i32) -> String {
        match index {
            0 => format!("{:.0}", self.cutoff.get()),
            1 => format!("{:.3}", 2. / self.res.get()),
            2 => format!("{:.2}", 20. * (self.drive.get() + 1.).log10()),
            3 => match self.mode.load(Ordering::Relaxed) {
                0 => format!("Lowpass"),
                1 => format!("Highpass"),
                2 => format!("Bandpass"),
                3 => format!("Notch"),
                4 => format!("Peak"),
                _ => format!("Peak"),
            },
            _ => format!(""),
        }
    }
}
impl Default for SVF {
    fn default() -> SVF {
        SVF {
            vout: [0f32; 2],
            s: [0f32; 2],
            params: Arc::new(FilterParameters::default()),
        }
    }
}
impl Plugin for SVF {
    fn set_sample_rate(&mut self, rate: f32) {
        self.params.sample_rate.set(rate);
    }
    fn get_info(&self) -> Info {
        Info {
            name: "SVF".to_string(),
            unique_id: 80371372,
            inputs: 1,
            outputs: 1,
            category: Category::Effect,
            parameters: 4,
            ..Default::default()
        }
    }
    // the DAW calls process every time a buffer of samples needs to be sent through the vst
    // buffer consists of both input and output buffers
    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        // split the buffer into input and output
        for (input_buffer, output_buffer) in buffer.zip() {
            // iterate through each sample in the input and output buffer
            for (input_sample, output_sample) in input_buffer.iter().zip(output_buffer) {
                // get the output sample by processing the input sample
                *output_sample = self.tick_pivotal(*input_sample);
            }
        }
    }
    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
}
plugin_main!(SVF);
