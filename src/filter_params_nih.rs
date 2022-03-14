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
                "Filter Frequency",
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
                "Filter Resonance",
                0.5,
                FloatRange::Linear { min: 0., max: 1. },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::f32_rounded(2))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),
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
            .with_value_to_string(formatters::f32_rounded(2)),

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

// pub struct FilterParameters {
//     pub g: AtomicF32,
//     pub sample_rate: AtomicF32,
//     pub zeta: AtomicF32,
//     pub k_ladder: AtomicF32,

//     // the "cutoff" parameter. Determines how heavy filtering is
//     pub cutoff: ParameterF32,
//     pub res: ParameterF32,

//     pub drive: ParameterF32,
//     pub mode: ParameterUsize,
//     pub slope: ParameterUsize,
//     pub filter_type: AtomicUsize,
// }
// impl FilterParameters {
//     // transform resonance parameter into something more useful for the 2 filters
//     pub fn set_resonances(&self) {
//         let res = self.res.get_normalized();
//         self.zeta.set(5. - 4.9 * res);
//         self.k_ladder.set(res.powi(2) * 3.8 - 0.2);
//     }
//     pub fn update_g(&self) {
//         self.g
//             .set((PI * self.cutoff.get() / (self.sample_rate.get())).tan());
//     }
//     pub fn get_parameter_default(&self, index: i32) -> f32 {
//         match index {
//             0 => self.cutoff.get_normalized_default(),
//             1 => self.res.get_normalized_default(),
//             2 => self.drive.get_normalized_default(),
//             3 => 0.,
//             4 => self.mode.get_normalized_default() as f32,
//             5 => self.slope.get_normalized_default() as f32,
//             _ => 0.0,
//         }
//     }
// }

// impl Default for FilterParameters {
//     fn default() -> FilterParameters {
//         let a = FilterParameters {
//             sample_rate: AtomicF32::new(48000.),
//             filter_type: AtomicUsize::new(0),
//             cutoff: ParameterF32::new(
//                 "Cutoff",
//                 10000.,
//                 0.,
//                 20000.,
//                 |x| format!("{:.0}Hz", x),
//                 |x| (1.8f32.powf(10. * x - 10.)),
//                 |x: f32| 1. + 0.17012975 * (x).ln(),
//             ),
//             g: AtomicF32::new((PI * 10000. / 48000.).tan()),

//             res: (ParameterF32::new(
//                 "Res",
//                 0.5,
//                 0.,
//                 1.,
//                 |x| format!("{:.2}%", x * 100.),
//                 |x: f32| x,
//                 |x: f32| x,
//             )),
//             drive: (ParameterF32::new(
//                 "Drive",
//                 0.,
//                 0.,
//                 14.8490,
//                 |x: f32| format!("{:.2}dB", 20. * (x + 1.).log10()),
//                 |x| x.powi(2),
//                 |x| x.sqrt(),
//             )),
//             mode: (ParameterUsize::new("Mode", 0, 0, 4, |x| match x {
//                 0 => format!("LP"),
//                 1 => format!("HP"),
//                 2 => format!("BP1"),
//                 3 => format!("Notch"),
//                 _ => format!("BP2"),
//             })),
//             slope: (ParameterUsize::new("Slope", 3, 0, 3, |x| match x {
//                 0 => format!("LP6"),
//                 1 => format!("LP12"),
//                 2 => format!("LP18"),
//                 3 => format!("LP24"),
//                 _ => format!("???"),
//             })),
//             k_ladder: AtomicF32::new(0.),
//             zeta: AtomicF32::new(0.),
//         };
//         a.g.set((PI * a.cutoff.get() / (a.sample_rate.get())).tan());
//         a.set_resonances();
//         a
//     }
// }

// impl PluginParameters for FilterParameters {
//     fn get_parameter(&self, index: i32) -> f32 {
//         match index {
//             0 => self.cutoff.get_normalized(),
//             1 => self.res.get_normalized(),
//             2 => self.drive.get_normalized(),
//             3 => self.filter_type.get() as f32,
//             4 => self.mode.get_normalized() as f32,
//             5 => self.slope.get_normalized() as f32,
//             _ => 0.0,
//         }
//     }
//     fn set_parameter(&self, index: i32, value: f32) {
//         match index {
//             0 => {
//                 self.cutoff.set_normalized(value);
//                 self.update_g();
//             }
//             1 => {
//                 self.res.set_normalized(value);
//                 self.set_resonances();
//             }
//             2 => self.drive.set_normalized(value),
//             // TODO: filter_type won't work with more than 2 filter modes, make proper param
//             3 => {
//                 self.filter_type.set(value as usize);
//             }
//             4 => self.mode.set_normalized(value),
//             5 => self.slope.set_normalized(value),
//             _ => (),
//         }
//     }
//     fn get_parameter_name(&self, index: i32) -> String {
//         match index {
//             0 => self.cutoff.get_name(),
//             1 => self.res.get_name(),
//             2 => self.drive.get_name(),
//             3 => "filter type".to_string(),
//             4 => self.mode.get_name(),
//             5 => self.slope.get_name(),
//             _ => "".to_string(),
//         }
//     }
//     // This is what will display underneath our control.  We can
//     // format it into a string that makes sense for the user.
//     fn get_parameter_text(&self, index: i32) -> String {
//         match index {
//             0 => self.cutoff.get_display(),
//             1 => self.res.get_display(),
//             2 => self.drive.get_display(),
//             3 => match self.filter_type.get() {
//                 0 => "State variable".to_string(),
//                 _ => "Transistor ladder".to_string(),
//             },
//             4 => self.mode.get_display(),
//             5 => self.slope.get_display(),
//             _ => format!(""),
//         }
//     }
//     // transforms the plugin state into a byte vector.
//     // For this plugin, this is just the parameters' normalized values
//     fn get_preset_data(&self) -> Vec<u8> {
//         // std::slice::from_raw_parts(data, len)
//         let mut param_vec = Vec::new();
//         // Remember to update n_params when adding more
//         let n_params = 6;
//         for i in 0..n_params {
//             param_vec.push(self.get_parameter(i));
//         }
//         let param_vec_u8 = bincode::serialize(&param_vec).unwrap();
//         param_vec_u8
//     }
//     // this should use a byte vec from the method above
//     fn load_preset_data(&self, data: &[u8]) {
//         let n_params = 6;
//         let param_data = &data[0..(n_params + 2) * 4];
//         let param_vec: Vec<f32> = bincode::deserialize(param_data).unwrap();
//         for i in 0..n_params {
//             self.set_parameter(i as i32, param_vec[i]);
//         }
//     }
//     // some hosts call the bank_data methods instead of the preset_data methods when saving/loading state
//     // therefore they need to be implemented, even if they just do the same as the preset_data methods
//     // this should use a byte vec from the method above
//     fn load_bank_data(&self, data: &[u8]) {
//         self.load_preset_data(data);
//     }
//     fn get_bank_data(&self) -> Vec<u8> {
//         self.get_preset_data()
//     }
// }
