//! This zero-delay feedback filter is based on a state variable filter.
//! It follows the following equations:
//!
//! Since we can't easily solve a nonlinear equation,
//! Mystran's fixed-pivot method is used to approximate the tanh() parts.
//! Quality can be improved a lot by oversampling a bit.
//! Damping feedback is antisaturated, so it doesn't disappear at high gains.

// look into successive over-relaxation, Gauss–Seidel method, just making a runge-kutta solver
// Brent's method seems the most promising so far. Could potentially replace inverse quadratic with newton's
// or possibly just a broyden method fallback, can't be bothered working much more on this lol: http://fabcol.free.fr/pdf/lectnotes5.pdf
// check if it's well-behaved without the pivotal guess, and how to make pivotal more similar to newton?

// TODO: 
// The simd-ified filters are for some reason much slower (not sure if twice as slow, which would be break-even point)
// Benchmark them and the non-simd filters


#[macro_use]
extern crate vst;
use filter::{LadderFilter, SVF};
use packed_simd::f32x4;
use std::sync::Arc;
use vst::buffer::AudioBuffer;
use vst::editor::Editor;
use vst::plugin::{Category, HostCallback, Info, Plugin, PluginParameters};

use vst::event::Event;
use vst::api::Events;

mod editor;
use editor::{EditorState, SVFPluginEditor};
mod parameter;
#[allow(dead_code)]
mod utils;
use utils::AtomicOps;
mod filter_parameters;
use filter_parameters::FilterParameters;

mod ui;
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
impl VST {
    fn process_midi_event(&self, data: [u8; 3]) {
        match data[0] {
            // controller change
            0xB0 => {
                // mod wheel
                if data[1] == 1 {
                    // TODO: Might want to use hostcallback to automate here
                    self.params.set_parameter(0, data[2] as f32 / 127.)
                }
            }
            _ => (),
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
                state: Arc::new(EditorState::new(params, Some(host))),
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
        // // split the buffer into input and output
        // let (inputs, outputs) = buffer.split();
        // // Iterate over inputs as (&f32, &f32)
        // let (l, r) = inputs.split_at(1);
        // let stereo_in = l[0].iter().zip(r[0].iter());
        // // Iterate over outputs as (&mut f32, &mut f32)
        // let (mut l, mut r) = outputs.split_at_mut(1);
        // let stereo_out = l[0].iter_mut().zip(r[0].iter_mut());
        
        // potentially the duplications of code could be hidden away with process_buffer functions
        if self.params.filter_type.get() == 0 {
            for (input_buffer, output_buffer) in buffer.zip() {
                // iterate through each sample in the input and output buffer
                for (input_sample, output_sample) in input_buffer.iter().zip(output_buffer) {
                    // get the output sample by processing the input sample
                    let frame = f32x4::new(*input_sample, 0.0, 0.0, 0.0);
                    // would be nice to align this, but doesn't seem possible with #[repr(align)].
                    // ah well. not much of a perf penalty for unaligned writes these days.
                    let processed = self.svf.tick_newton(frame);

                    let mut frame_out = [0.0f32; 4];
                    unsafe {
                        processed.write_to_slice_unaligned_unchecked(&mut frame_out);
                    }
                    // get the output sample by processing the input sample
                    *output_sample = frame_out[0];
                }
            }
        } else {
            for (input_buffer, output_buffer) in buffer.zip() {
                // iterate through each sample in the input and output buffer
                for (input_sample, output_sample) in input_buffer.iter().zip(output_buffer) {

                    // let frame = f32x4::new(input[0][i], input[1][i], 0.0, 0.0);
                    let frame = f32x4::new(*input_sample, 0.0, 0.0, 0.0);
                    // would be nice to align this, but doesn't seem possible with #[repr(align)].
                    // ah well. not much of a perf penalty for unaligned writes these days.
                    let processed = self.ladder.tick_newton(frame);

                    let mut frame_out = [0.0f32; 4];
                    unsafe {
                        processed.write_to_slice_unaligned_unchecked(&mut frame_out);
                    }
                    // get the output sample by processing the input sample
                    *output_sample = frame_out[0];
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
    // handling of midi events
    fn process_events(&mut self, events: &Events) {
        for event in events.events() {
            match event {
                Event::Midi(ev) => self.process_midi_event(ev.data),
                // More events can be handled here.
                _ => (),
            }
        }
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
