#![feature(portable_simd)]
use core_simd::simd::f32x4;
use filter::{preprocess, LadderFilter};

use std::sync::Arc;

use nih_plug::{nih_export_vst3, prelude::*};

mod editor;
use editor::*;
pub mod utils;
use utils::AtomicOps;
pub mod filter_params;
use filter_params::FilterParams;

mod resampling;
use resampling::HalfbandFilter;

pub mod filter;
mod ui;

pub struct VaFilter {
    // Store a handle to the plugin's parameter object.
    params: Arc<FilterParams>,
    ladder: filter::LadderFilter,

    svf_stereo: filter::svf::Svf,
    sallenkey_stereo: filter::sallen_key::SallenKey,

    should_update_filter: Arc<std::sync::atomic::AtomicBool>,

    upsampler: HalfbandFilter,
    downsampler: HalfbandFilter,
    dc_filter: preprocess::DcFilter,

    oversample_factor: usize,
}

impl Default for VaFilter {
    fn default() -> Self {
        let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let params = Arc::new(FilterParams::new(should_update_filter.clone()));

        let ladder = LadderFilter::new(params.clone());
        let svf_stereo = filter::svf::Svf::new(params.clone());
        let sallenkey_stereo = filter::sallen_key::SallenKey::new(params.clone());

        Self {
            params,
            should_update_filter,

            svf_stereo,
            sallenkey_stereo,
            ladder,

            upsampler: HalfbandFilter::new(8, true),
            downsampler: HalfbandFilter::new(8, true),
            dc_filter: preprocess::DcFilter::default(),
            oversample_factor: 2,
        }
    }
}

impl Plugin for VaFilter {
    const NAME: &'static str = "Va Filter";
    const VENDOR: &'static str = "???";
    const URL: &'static str = "github.com/fredemus/va-filter";
    const EMAIL: &'static str = "???";

    const VERSION: &'static str = "0.0.1";

    const DEFAULT_INPUT_CHANNELS: u32 = 2;
    const DEFAULT_OUTPUT_CHANNELS: u32 = 2;

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
        _context: &mut impl InitContext,
    ) -> bool {
        let fs = _buffer_config.sample_rate;
        if fs >= 88200. {
            self.params.sample_rate.set(fs);
            self.oversample_factor = 1;
        } else {
            self.params.sample_rate.set(2. * fs);
            self.oversample_factor = 2;
        }
        true
    }
    fn reset(&mut self) {
        self.sallenkey_stereo.reset();
        self.svf_stereo.reset();
        self.ladder.s = [f32x4::splat(0.); 4];
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext,
    ) -> ProcessStatus {
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
            self.params.update_g(self.params.cutoff.value());
            self.params.set_resonances(self.params.res.value());

            self.sallenkey_stereo.update();
            self.svf_stereo.update();
        }
        for mut channel_samples in buffer.iter_samples() {
            if self.params.cutoff.smoothed.is_smoothing() {
                let cut_smooth = self.params.cutoff.smoothed.next();
                self.params.update_g(cut_smooth);

                self.sallenkey_stereo.update();
                self.svf_stereo.update();
            }
            if self.params.res.smoothed.is_smoothing() {
                let res_smooth = self.params.res.smoothed.next();
                self.params.set_resonances(res_smooth);

                self.sallenkey_stereo.update();
                self.svf_stereo.update();
            }

            let in_l = *channel_samples.get_mut(0).unwrap();
            let in_r = *channel_samples.get_mut(1).unwrap();
            let mut frame = f32x4::from_array([in_l, in_r, 0.0, 0.0]);

            // filter before oversampling to remove dc-offset, since offsets can make the models behave weirdly
            frame = self.dc_filter.process(frame);

            let processed;
            if self.oversample_factor == 2 {
                // zero-stuff input
                let input = [frame, f32x4::splat(0.)];
                let mut output = f32x4::splat(0.);
                for i in 0..2 {
                    // run input audio through a half-band filter
                    // multiply by oversample factor (2) to avoid the volume loss from zero-stuffing
                    let frame = self.upsampler.process(f32x4::splat(2.) * input[i]);

                    // perform filtering with the cool filters
                    let filter_out = match self.params.filter_type.value() {
                        // filter_params_nih::Circuits::SVF => self.svf.tick_newton(frame),
                        filter_params::Circuits::SallenKey => self.sallenkey_stereo.process(frame),
                        filter_params::Circuits::SVF => self.svf_stereo.process(frame),
                        _ => self.ladder.tick_newton(frame),
                    };

                    // downsample filter, removing frequencies above nyquist
                    output = self.downsampler.process(filter_out);
                }
                processed = output;
            } else {
                processed = match self.params.filter_type.value() {
                    filter_params::Circuits::SallenKey => self.sallenkey_stereo.process(frame),
                    filter_params::Circuits::SVF => self.svf_stereo.process(frame),
                    _ => self.ladder.tick_newton(frame),
                };
            }
            let frame_out = *processed.as_array();
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

impl ClapPlugin for VaFilter {
    const CLAP_ID: &'static str = "https://github.com/Fredemus/va-filter/";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Va filter");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

nih_export_vst3!(VaFilter);
nih_export_clap!(VaFilter);
