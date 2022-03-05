use vst::plugin::PluginParameters;

use crate::parameter::Parameter;

use super::parameter::{ParameterF32, ParameterUsize};
use super::utils::*;
use enum_index::{EnumIndex, IndexEnum};

use enum_index_derive::{EnumIndex, IndexEnum};

use std::fmt;

use std::f32::consts::PI;

pub struct FilterParameters {
    pub g: AtomicF32,
    pub sample_rate: AtomicF32,

    // the "cutoff" parameter. Determines how heavy filtering is
    pub cutoff: ParameterF32,
    pub res: ParameterF32,
    pub zeta: AtomicF32,
    pub k_ladder: AtomicF32,

    pub drive: ParameterF32,
    pub mode: ParameterUsize,
    pub slope: ParameterUsize,
    pub filter_type: ParameterUsize,
}

#[repr(C)]
#[derive(EnumIndex, IndexEnum, Debug)]
pub enum FilterParameterNr {
    Cutoff,
    Res,
    Drive,
    FilterType,
    Mode,
    Slope,
}

#[repr(C)]
#[derive(EnumIndex, IndexEnum, Debug)]
pub enum Mode {
    LP,
    HP,
    BP1,
    Notch,
    BP2,
}

#[repr(C)]
#[derive(EnumIndex, IndexEnum, Debug)]
pub enum Slope {
    LP6,
    LP12,
    LP18,
    LP24,
}

#[repr(C)]
#[derive(EnumIndex, IndexEnum)]
pub enum FilterType {
    StateVariableFilter,
    TransistorLadderFilter,
}

impl fmt::Debug for FilterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                FilterType::StateVariableFilter => &"State Variable Filter",
                FilterType::TransistorLadderFilter => &"Transistor Ladder",
            }
        )
    }
}

impl FilterParameters {
    // transform resonance parameter into something more useful for the 2 filters
    pub fn set_resonances(&self) {
        let res = self.res.get_normalized();
        self.zeta.set(5. - 4.9 * res);
        self.k_ladder.set(res.powi(2) * 3.8 - 0.2);
    }
    pub fn update_g(&self) {
        self.g
            .set((PI * self.cutoff.get() / (self.sample_rate.get())).tan());
    }

    pub fn get_parameter_default(&self, index: i32) -> f32 {
        match FilterParameterNr::index_enum(index as usize).unwrap() {
            Cutoff => self.cutoff.get_normalized_default(),
            Res => self.res.get_normalized_default(),
            Drive => self.drive.get_normalized_default(),
            FilterType => self.filter_type.get_normalized_default(),
            Mode => self.mode.get_normalized_default(),
            Slope => self.slope.get_normalized_default(),
        }
    }
}

impl Default for FilterParameters {
    fn default() -> FilterParameters {
        let filter_parameters = FilterParameters {
            sample_rate: AtomicF32::new(48000.),

            cutoff: ParameterF32::new(
                "Cutoff",
                10000.,
                0.,
                20000.,
                |x| format!("{:.0}Hz", x),
                |x| (1.8f32.powf(10. * x - 10.)),
                |x: f32| 1. + 0.17012975 * (x).ln(),
            ),
            g: AtomicF32::new((PI * 10000. / 48000.).tan()),

            res: (ParameterF32::new(
                "Res",
                0.5,
                0.,
                1.,
                |x| format!("{:.2}%", x * 100.),
                |x: f32| x,
                |x: f32| x,
            )),
            drive: (ParameterF32::new(
                "Drive",
                0.,
                0.,
                14.8490,
                |x: f32| format!("{:.2}dB", 20. * (x + 1.).log10()),
                |x| x.powi(2),
                |x| x.sqrt(),
            )),

            filter_type: (ParameterUsize::new("Type", 0, 0, 1, |x| {
                format!("{:?}", FilterType::index_enum(x).unwrap())
            })),

            mode: (ParameterUsize::new("Mode", 0, 0, 4, |x| {
                format!("{:?}", Mode::index_enum(x).unwrap())
            })),

            slope: (ParameterUsize::new("Slope", 3, 0, 3, |x| {
                format!("{:?}", Slope::index_enum(x).unwrap())
            })),

            k_ladder: AtomicF32::new(0.),
            zeta: AtomicF32::new(0.),
        };
        filter_parameters.g.set(
            (PI * filter_parameters.cutoff.get() / (filter_parameters.sample_rate.get())).tan(),
        );
        filter_parameters.set_resonances();
        filter_parameters
    }
}

use FilterParameterNr::*;
impl PluginParameters for FilterParameters {
    fn get_parameter(&self, index: i32) -> f32 {
        match FilterParameterNr::index_enum(index as usize).unwrap() {
            Cutoff => self.cutoff.get_normalized(),
            Res => self.res.get_normalized(),
            Drive => self.drive.get_normalized(),
            FilterType => self.filter_type.get() as f32,
            Mode => self.mode.get_normalized() as f32,
            Slope => self.slope.get_normalized() as f32,
        }
    }
    fn set_parameter(&self, index: i32, value: f32) {
        match FilterParameterNr::index_enum(index as usize).unwrap() {
            Cutoff => {
                self.cutoff.set_normalized(value);
                self.update_g();
            }
            Res => {
                self.res.set_normalized(value);
                self.set_resonances();
            }
            Drive => self.drive.set_normalized(value),
            FilterType => {
                self.filter_type.set_normalized(value);
            }
            Mode => self.mode.set_normalized(value),
            Slope => self.slope.set_normalized(value),
        }
    }
    fn get_parameter_name(&self, index: i32) -> String {
        match FilterParameterNr::index_enum(index as usize).unwrap() {
            Cutoff => self.cutoff.get_name(),
            Res => self.res.get_name(),
            Drive => self.drive.get_name(),
            FilterType => self.filter_type.get_name(),
            Mode => self.mode.get_name(),
            Slope => self.slope.get_name(),
        }
    }
    // This is what will display underneath our control.  We can
    // format it into a string that makes sense for the user.
    fn get_parameter_text(&self, index: i32) -> String {
        match FilterParameterNr::index_enum(index as usize).unwrap() {
            Cutoff => self.cutoff.get_display(),
            Res => self.res.get_display(),
            Drive => self.drive.get_display(),
            FilterType => self.filter_type.get_display(),
            Mode => self.mode.get_display(),
            Slope => self.slope.get_display(),
        }
    }
    // transforms the plugin state into a byte vector.
    // For this plugin, this is just the parameters' normalized values
    fn get_preset_data(&self) -> Vec<u8> {
        // std::slice::from_raw_parts(data, len)
        let mut param_vec = Vec::new();
        // Remember to update n_params when adding more
        let n_params = 6;
        for i in 0..n_params {
            param_vec.push(self.get_parameter(i));
        }
        let param_vec_u8 = bincode::serialize(&param_vec).unwrap();
        param_vec_u8
    }
    // this should use a byte vec from the method above
    fn load_preset_data(&self, data: &[u8]) {
        let n_params = 6;
        let param_data = &data[0..(n_params + 2) * 4];
        let param_vec: Vec<f32> = bincode::deserialize(param_data).unwrap();
        for i in 0..n_params {
            self.set_parameter(i as i32, param_vec[i]);
        }
    }
    // some hosts call the bank_data methods instead of the preset_data methods when saving/loading state
    // therefore they need to be implemented, even if they just do the same as the preset_data methods
    // this should use a byte vec from the method above
    fn load_bank_data(&self, data: &[u8]) {
        self.load_preset_data(data);
    }
    fn get_bank_data(&self) -> Vec<u8> {
        self.get_preset_data()
    }
}
