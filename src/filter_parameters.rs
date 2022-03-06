use num_enum::{FromPrimitive, IntoPrimitive};
use strum::{EnumIter, IntoEnumIterator};
use vst::plugin::PluginParameters;

use crate::parameter::Parameter;

use super::parameter::{ParameterF32, ParameterUsize};
use super::utils::*;
use crate::parameter::GetParameterByIndex;

use std::fmt;

use std::convert::Into;
use std::f32::consts::PI;

// Helper for the indexing, could be moved elsewhere
trait MaxIndex {
    fn max_index() -> i32;
}

impl<T> MaxIndex for T
where
    T: IntoEnumIterator + Into<i32>,
{
    fn max_index() -> i32 {
        T::iter().fold(0, |max_index, v| max_index.max(v.into()))
    }
}

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

    pub reserved: ParameterUsize,
}

#[repr(i32)]
#[derive(FromPrimitive, IntoPrimitive, Eq, PartialEq, Debug, EnumIter)]
pub enum Parameters {
    #[num_enum(default)]
    Cutoff,
    Res,
    Drive,

    FilterType,
    Mode,
    Slope,
}

impl GetParameterByIndex for FilterParameters {
    fn get_parameter_by_index<'a>(&'a self, index: i32) -> &'a dyn Parameter {
        match Parameters::from(index) {
            Cutoff => &self.cutoff,
            Res => &self.res,
            Drive => &self.drive,
            FilterType => &self.filter_type,
            Mode => &self.mode,
            Slope => &self.slope,
        }
    }
}

#[repr(i32)]
#[derive(FromPrimitive, Eq, PartialEq, Debug)]
pub enum Mode {
    #[num_enum(default)]
    LP,
    HP,
    BP1,
    Notch,
    BP2,
}

#[repr(i32)]
#[derive(FromPrimitive, Eq, PartialEq, Debug)]
pub enum Slope {
    #[num_enum(default)]
    LP6,
    LP12,
    LP18,
    LP24,
}

#[repr(i32)]
#[derive(FromPrimitive, Eq, PartialEq)]
pub enum FilterType {
    #[num_enum(default)]
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
                format!("{:?}", FilterType::from(x as i32))
            })),

            mode: (ParameterUsize::new("Mode", 0, 0, 4, |x| format!("{:?}", Mode::from(x as i32)))),

            slope: (ParameterUsize::new("Slope", 3, 0, 3, |x| {
                format!("{:?}", Slope::from(x as i32))
            })),

            reserved: (ParameterUsize::new("Reserved", 0, 0, 1, |_| "reserved".to_string())),

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

use Parameters::*;
impl PluginParameters for FilterParameters {
    fn get_parameter(&self, index: i32) -> f32 {
        self.get_parameter_by_index(index).get_normalized()
    }

    fn set_parameter(&self, index: i32, value: f32) {
        match Parameters::from(index) {
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
            Padding => self.reserved.set_normalized(value),
        }
    }
    fn get_parameter_name(&self, index: i32) -> String {
        self.get_parameter_by_index(index).get_name()
    }
    // This is what will display underneath our control.  We can
    // format it into a string that makes sense for the user.
    fn get_parameter_text(&self, index: i32) -> String {
        self.get_parameter_by_index(index).get_display()
    }
    // transforms the plugin state into a byte vector.
    // For this plugin, this is just the parameters' normalized values
    fn get_preset_data(&self) -> Vec<u8> {
        // std::slice::from_raw_parts(data, len)
        let mut param_vec = Vec::new();

        let max_index = Parameters::max_index();
        for i in 0..=max_index {
            param_vec.push(self.get_parameter(i));
        }
        let param_vec_u8 = bincode::serialize(&param_vec).unwrap();
        param_vec_u8
    }
    // this should use a byte vec from the method above
    fn load_preset_data(&self, data: &[u8]) {
        let max_index = Parameters::max_index() as usize;

        let param_data = data;
        let param_vec: Vec<f32> = bincode::deserialize(param_data).unwrap();
        for i in 0..=max_index {
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

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_index() {
        let filter_parameters = FilterParameters::default();

        for i in 0..5 {
            println!(" {:?}", filter_parameters.get_parameter_text(i));
        }
    }

    #[test]
    fn test_get_preset() {
        use super::*;
        let filter_parameters = FilterParameters::default();

        let slope = filter_parameters.get_parameter_by_index(Parameters::Slope as i32);

        println!(
            "Slope.  {} {} {}",
            slope.get_name(),
            slope.get_display(),
            slope.get_normalized()
        );

        slope.set_normalized(0.3);

        println!(
            "Slope.  {} {} {}",
            slope.get_name(),
            slope.get_display(),
            slope.get_normalized()
        );

        let data = filter_parameters.get_preset_data();
        println!("data {:?}", data);

        let load = FilterParameters::default();

        load.load_preset_data(&data);
        let load_slope = load.get_parameter_by_index(Parameters::Slope as i32);

        println!(
            "Slope.  {} {} {}",
            load_slope.get_name(),
            load_slope.get_display(),
            load_slope.get_normalized()
        );
        assert! { load_slope.get_normalized() == 0.3 }
    }

    #[test]
    fn test_max_index() {
        use super::*;

        #[repr(i32)]
        #[derive(FromPrimitive, IntoPrimitive, Eq, PartialEq, Debug, EnumIter)]
        enum Test {
            #[num_enum(default)]
            Cutoff,
            Res,
            Drive,

            FilterType,
            Mode,
            Slope,

            End = 200,
        }

        let max_raw = Test::iter().fold(0, |max_index, v| max_index.max(v as i32));

        println!("max_raw  {}", max_raw);

        let max_trait = Test::max_index();

        println!("max_trait  {}", max_trait);

        assert_eq!(max_raw, 200);
        assert_eq!(max_trait, 200);
    }
}
