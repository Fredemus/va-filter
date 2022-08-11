#![feature(portable_simd)]
// #[macro_use]
// extern crate vst;
use filter::LadderFilter;
// use packed_simd::f32x4;
use core_simd::f32x4;
// use vst::buffer::AudioBuffer;
// use vst::editor::Editor;
// use vst::plugin::{CanDo, Category, HostCallback, Info, Plugin, PluginParameters};

// use vst::api::Events;
// use vst::event::Event;
use std::sync::Arc;

use nih_plug::{nih_export_vst3, prelude::*};

// mod editor;
// use editor::{EditorState, SVFPluginEditor};
mod editor;
use editor::*;
mod parameter;
#[allow(dead_code)]
mod utils;
use utils::AtomicOps;
mod filter_params_nih;
use filter_params_nih::FilterParams;

mod filter;
mod ui;

pub(crate) struct VaFilter {
    // Store a handle to the plugin's parameter object.
    params: Arc<FilterParams>,
    ladder: filter::LadderFilter,
    // svf: filter::SVF,
    svf_new: filter::Svf,

    sallenkey: filter::SallenKey,
    // used for constructing the editor in get_editor
    // host: Option<HostCallback>,
    /// If this is set at the start of the processing cycle, then the filter coefficients should be
    /// updated. For the regular filter parameters we can look at the smoothers, but this is needed
    /// when changing the number of active filters.
    should_update_filter: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for VaFilter {
    fn default() -> Self {
        let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let params = Arc::new(FilterParams::new(should_update_filter.clone()));
        // let svf = SVF::new(params.clone());
        let sallenkey = filter::SallenKey::new(params.clone());
        let svf_new = filter::Svf::new(params.clone());
        let ladder = LadderFilter::new(params.clone());
        Self {
            params: params.clone(),
            // svf,
            svf_new,
            sallenkey,
            ladder,
            should_update_filter,
            // host: None,
        }
    }
}

impl Plugin for VaFilter {
    const NAME: &'static str = "Va Filter";
    const VENDOR: &'static str = "???";
    const URL: &'static str = "???";
    const EMAIL: &'static str = "???";

    const VERSION: &'static str = "0.0.1";

    const DEFAULT_NUM_INPUTS: u32 = 2;
    const DEFAULT_NUM_OUTPUTS: u32 = 2;

    const MIDI_INPUT: MidiConfig = MidiConfig::None;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&self) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();

        create_vizia_editor(move |cx, context| {
            ui::plugin_gui(cx, params.clone(), context.clone());
        })
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        // This works with any symmetrical IO layout
        config.num_input_channels == config.num_output_channels && config.num_input_channels > 0
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        _buffer_config: &BufferConfig,
        _context: &mut impl ProcessContext,
    ) -> bool {
        self.params.sample_rate.set(_buffer_config.sample_rate);
        true
    }
    fn reset(&mut self) {
        self.sallenkey.s = [0.; 2];
        self.svf_new.s = [0.; 2];
        self.ladder.s = [f32x4::splat(0.); 4];
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _context: &mut impl ProcessContext,
    ) -> ProcessStatus {
        for mut channel_samples in buffer.iter_samples() {
            if self
                .should_update_filter
                .compare_exchange(
                    true,
                    false,
                    std::sync::atomic::Ordering::Acquire,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                self.params.update_g(self.params.cutoff.value);
                self.params.set_resonances(self.params.res.value);
            }
            if self.params.cutoff.smoothed.is_smoothing() {
                let cut_smooth = self.params.cutoff.smoothed.next();
                self.params.update_g(cut_smooth);
                self.sallenkey.update_matrices();
                self.svf_new.update_matrices();
            }
            if self.params.res.smoothed.is_smoothing() {
                let res_smooth = self.params.res.smoothed.next();
                self.params.set_resonances(res_smooth);
                self.sallenkey.update_matrices();
                self.svf_new.update_matrices();
            }

            // channel_samples[0];
            let frame = f32x4::from_array([
                *channel_samples.get_mut(0).unwrap(),
                *channel_samples.get_mut(1).unwrap(),
                0.0,
                0.0,
            ]);
            // let mut samples = unsafe { channel_samples.to_simd_unchecked() };
            let processed = match self.params.filter_type.value() {
                // filter_params_nih::Circuits::SVF => self.svf.tick_newton(frame),
                filter_params_nih::Circuits::SallenKey => self.sallenkey.tick_dk(frame),
                filter_params_nih::Circuits::SVF => self.svf_new.tick_dk(frame),
                filter_params_nih::Circuits::Ladder => self.ladder.tick_newton(frame),
            };

            // let processed = self.ladder.tick_linear(frame);
            let frame_out = *processed.as_array();
            // let frame_out = *frame.as_array();
            *channel_samples.get_mut(0).unwrap() = frame_out[0];
            *channel_samples.get_mut(1).unwrap() = frame_out[1];
        }

        ProcessStatus::Normal
    }
}

impl Vst3Plugin for VaFilter {
    const VST3_CLASS_ID: [u8; 16] = *b"Va-filter       ";
    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_vst3!(VaFilter);
