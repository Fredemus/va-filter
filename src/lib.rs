//! This zero-delay feedback filter is based on a state variable filter.
//! It follows the following equations:
//!
//! Since we can't easily solve a nonlinear equation,
//! Mystran's fixed-pivot method is used to approximate the tanh() parts.
//! Quality can be improved a lot by oversampling a bit.
//! Damping feedback is antisaturated, so it doesn't disappear at high gains.

// TODO:
// look into successive over-relaxation, Gauss–Seidel method, just making a runge-kutta solver
// Brent's method seems the most promising so far. Could potentially replace inverse quadratic with newton's
// or possibly just a broyden method fallback, can't be bothered working much more on this lol: http://fabcol.free.fr/pdf/lectnotes5.pdf
// check if it's well-behaved without the pivotal guess, and how to make pivotal more similar to newton?

#[macro_use]
extern crate vst;
use filter::{LadderFilter, SVF};
use std::f32::consts::PI;
use std::sync::Arc;
use vst::buffer::AudioBuffer;
use vst::editor::Editor;
use vst::plugin::{Category, HostCallback, Info, Plugin, PluginParameters};

mod editor;
use editor::{EditorState, SVFPluginEditor};
mod parameter;
#[allow(dead_code)]
mod utils;
use utils::AtomicOps;
mod filter_parameters;
use filter_parameters::FilterParameters;

mod filter;

// this is a 2-pole filter with resonance, which is why there's 2 states and vouts
struct VST {
    // Store a handle to the plugin's parameter object.
    params: Arc<FilterParameters>,
    // The object responsible for the gui
    editor: Option<SVFPluginEditor>,

    ladder: filter::LadderFilter,
    svf: filter::SVF,
}
impl VST {}
impl FilterParameters {
    pub fn update_g(&self) {
        self.g
            .set((PI * self.cutoff.get() / (self.sample_rate.get())).tan());
    }
}
impl PluginParameters for FilterParameters {
    fn get_parameter(&self, index: i32) -> f32 {
        match index {
            0 => self.cutoff.get_normalized(),
            1 => self.res.get_normalized(),
            2 => self.drive.get_normalized(),
            3 => self.filter_type.get() as f32,
            4 => self.mode.get_normalized() as f32,
            5 => self.slope.get_normalized() as f32,
            _ => 0.0,
        }
    }
    fn set_parameter(&self, index: i32, value: f32) {
        match index {
            0 => {
                self.cutoff.set_normalized(value);
                self.update_g();
            }
            1 => {
                self.res.set_normalized(value);
                self.set_resonances();
            }
            2 => self.drive.set_normalized(value),
            // TODO: filter_type won't work with more than 2 filter modes, make proper param
            3 => {
                self.filter_type.set(value as usize);
            }
            4 => self.mode.set_normalized(value),
            5 => self.slope.set_normalized(value),
            _ => (),
        }
    }
    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            0 => "cutoff".to_string(),
            1 => "resonance".to_string(),
            2 => "drive".to_string(),
            3 => "filter type".to_string(),
            4 => "filter mode".to_string(),
            5 => "filter slope".to_string(),
            _ => "".to_string(),
        }
    }
    // This is what will display underneath our control.  We can
    // format it into a string that makes sense for the user.
    fn get_parameter_text(&self, index: i32) -> String {
        match index {
            0 => self.cutoff.get_display(),
            1 => self.res.get_display(),
            2 => self.drive.get_display(),
            3 => match self.filter_type.get() {
                0 => "State variable".to_string(),
                _ => "Transistor ladder".to_string(),
            },
            4 => self.mode.get_display(),
            5 => self.slope.get_display(),
            _ => format!(""),
        }
    }
}
impl Default for VST {
    fn default() -> Self {
        let params = Arc::new(FilterParameters::default());
        let mut svf = SVF::default();
        svf.params = params.clone();
        let mut ladder = LadderFilter::default();
        ladder.params = params.clone();
        Self {
            params: params.clone(),
            editor: Some(SVFPluginEditor {
                is_open: false,
                state: Arc::new(EditorState {
                    params: params,
                    host: None,
                }),
            }),
            svf,
            ladder,
        }
    }
}
impl Plugin for VST {
    fn new(host: HostCallback) -> Self {
        let params = Arc::new(FilterParameters::default());
        let mut svf = SVF::default();
        svf.params = params.clone();
        let mut ladder = LadderFilter::default();
        ladder.params = params.clone();
        Self {
            params: params.clone(),
            editor: Some(SVFPluginEditor {
                is_open: false,
                state: Arc::new(EditorState {
                    params: params,
                    host: Some(host),
                }),
            }),
            svf,
            ladder,
        }
    }
    fn set_sample_rate(&mut self, rate: f32) {
        self.params.sample_rate.set(rate);
        self.params.update_g();
    }
    fn get_info(&self) -> Info {
        Info {
            name: "SVF".to_string(),
            unique_id: 80371372,
            inputs: 1,
            outputs: 1,
            category: Category::Effect,
            parameters: 6,
            ..Default::default()
        }
    }
    // the DAW calls process every time a buffer of samples needs to be sent through the vst
    // buffer consists of both input and output buffers
    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        // split the buffer into input and output
        if self.params.filter_type.get() == 0 {
            for (input_buffer, output_buffer) in buffer.zip() {
                // iterate through each sample in the input and output buffer
                for (input_sample, output_sample) in input_buffer.iter().zip(output_buffer) {
                    // get the output sample by processing the input sample
                    // *output_sample = self.tick_pivotal(*input_sample);
                    // *output_sample = self.ladder.tick_newton(*input_sample);
                    *output_sample = self.svf.tick_newton(*input_sample);
                }
            }
        } else {
            for (input_buffer, output_buffer) in buffer.zip() {
                // iterate through each sample in the input and output buffer
                for (input_sample, output_sample) in input_buffer.iter().zip(output_buffer) {
                    // get the output sample by processing the input sample
                    *output_sample = self.ladder.tick_newton(*input_sample);
                }
            }
        }
    }
    fn get_editor(&mut self) -> Option<Box<dyn Editor>> {
        if let Some(editor) = self.editor.take() {
            Some(Box::new(editor) as Box<dyn Editor>)
        } else {
            None
        }
    }
    // lets the plugin host get access to the parameters
    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
}
plugin_main!(VST);

#[test]
fn dumbtest() {
    if true && 0 < 9 {
        println!("stuff makes sense");
    } else {
        println!("????")
    }
}
