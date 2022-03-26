// use vst::plugin::PluginParameters;

// use crate::parameter::Parameter;

// use super::parameter::{ParameterF32, ParameterUsize};
use super::utils::*;
use std::{f32::consts::PI, sync::atomic::AtomicBool};
// use std::sync::atomic::Ordering;
use nih_plug::prelude::*;
use std::{pin::Pin, sync::Arc};

#[derive(Params)]
pub struct FilterParams {
    #[id = "cutoff"]
    pub cutoff: FloatParam,
    #[id = "res"]
    pub res: FloatParam,
    #[id = "drive"]
    pub drive: FloatParam,

    #[id = "mode"]
    pub mode: EnumParam<SvfMode>,

    #[id = "slope"]
    pub slope: EnumParam<LadderSlope>,

    #[id = "circuit"]
    pub filter_type: EnumParam<Circuits>,

    pub g: AtomicF32,
    pub sample_rate: AtomicF32,
    pub zeta: AtomicF32,
    pub k_ladder: AtomicF32,
}

impl FilterParams {
    pub fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        let a = Self {
            // Smoothed parameters don't need the callback as we can just look at whether the
            // smoother is still smoothing
            // TODO: Need a callback here I think to update g?
            cutoff: FloatParam::new(
                "Cutoff",
                1000.0,
                FloatRange::Skewed {
                    min: 5.0, // This must never reach 0
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            // This needs quite a bit of smoothing to avoid artifacts
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" Hz")
            .with_value_to_string(formatters::f32_rounded(0))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),

            // TODO: Need a callback here I think to update q and res?
            res: FloatParam::new(
                "Res",
                0.5,
                FloatRange::Linear { min: 0., max: 1. },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::f32_rounded(2))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),
            // TODO: with_value_to_string should actually convert it to db
            drive: FloatParam::new(
                "Drive",
                0.0,
                FloatRange::Skewed {
                    min: 0.0, // This must never reach 0
                    max: 14.8490,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            // This needs quite a bit of smoothing to avoid artifacts
            .with_smoother(SmoothingStyle::Logarithmic(100.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::from_f32_gain_to_db(2)),

            mode: EnumParam::new("Mode", SvfMode::LP),
            // .with_callback(Arc::new(move |_| {
            //     should_update_filters.store(true, Ordering::Release)
            // })),
            slope: EnumParam::new("Slope", LadderSlope::LP24),

            filter_type: EnumParam::new("Filter type", Circuits::Ladder),

            k_ladder: AtomicF32::new(0.),
            zeta: AtomicF32::new(0.),
            g: AtomicF32::new(0.),
            sample_rate: AtomicF32::new(48000.),
        };
        a.update_g(a.cutoff.value);
        a.set_resonances(a.res.value);
        a
    }
    pub fn set_resonances(&self, val: f32) {
        let res = val;
        self.zeta.set(5. - 4.9 * res);
        self.k_ladder.set(res.powi(2) * 3.8 - 0.2);
    }
    pub fn update_g(&self, val: f32) {
        self.g.set((PI * val / (self.sample_rate.get())).tan());
    }
    // pub fn get_param<P: Param>(&self, index: usize) -> Box<dyn Param<Plain = f32>>
    // {
    //     match index {
    //         0 => &self.cutoff,

    //         _ => &self.slope,
    //     }
    // }
}

#[derive(Enum, Debug, PartialEq)]
pub enum SvfMode {
    LP,
    HP,
    BP1,
    Notch,
    BP2,
}
#[derive(Enum, Debug, PartialEq)]
pub enum LadderSlope {
    LP6,
    LP12,
    LP18,
    LP24,
}
#[derive(Enum, Debug, PartialEq)]
pub enum Circuits {
    SVF,
    Ladder,
}
