use super::parameter::Parameter;
use super::utils::*;
use std::f32::consts::PI;

pub struct FilterParameters {
    // the "cutoff" parameter. Determines how heavy filtering is
    // cutoff: AtomicFloat,
    // cutoff: Parameter<AtomicF32>,
    pub g: AtomicF32,
    pub sample_rate: AtomicF32,

    pub cutoff: Parameter<AtomicF32>,
    pub res: Parameter<AtomicF32>,
    pub drive: Parameter<AtomicF32>,
    pub mode: Parameter<AtomicUsize>,
    // cutoff: Params,
    // res: Params,
    // drive: Params,
    // mode: Params,
}
// use std::ops::Index;
// pub enum Params {
//     Usize(Parameter<AtomicUsize>),
//     F32(Parameter<AtomicF32>),
// }
// impl Params {
//     pub fn as_f32(&self) -> Option<&Parameter<AtomicF32>> {
//         match *self {
//             Params::F32(ref d) => Some(d),
//             _ => None,
//         }
//     }
//     pub fn as_usize(&self) -> Option<&Parameter<AtomicUsize>> {
//         match *self {
//             Params::Usize(ref d) => Some(d),
//             _ => None,
//         }
//     }
// }
// impl Index<usize> for FilterParameters
// {
//     type Output = Params;
//     fn index(&self, i: usize) -> &Self::Output {
//         match i {
//             //TODO: What's the best way to handle the reference here? removed for now i guess
//             // 0 => Params::F32(&self.cutoff),
//             0 => &self.cutoff,
//             1 => &self.res,
//             2 => &self.drive,
//             3 => &self.mode,
//             _ => &self.mode,
//         }
//     }
// }

impl Default for FilterParameters {
    fn default() -> FilterParameters {
        let a = FilterParameters {
            sample_rate: AtomicF32::new(48000.),
            cutoff: (Parameter::new("Cutoff", 20000., 0., 20000., |x| format!("{:.0} Hz", x), )),
            g: AtomicF32::new(0.),

            res: (Parameter::new("Resonance", 2. / 0.707, 20., 0.001, |x| {
                format!("{:.2}", 2. / x)
            })),
            drive: (Parameter::new("Drive", 0., 0., 20., |x: f32| {
                format!("{:.2} dB", 20. * (x + 1.).log10())
            })),
            mode: (Parameter::new("Filter mode", 0, 0, 4, |x| match x {
                0 => format!("Lowpass"),
                1 => format!("Highpass"),
                2 => format!("Bandpass"),
                3 => format!("Notch"),
                4 => format!("Peak"),
                _ => format!("Peak"),
            })),
        };
        a.g.set((PI * a.cutoff.get() / (a.sample_rate.get())).tan());
        return a;
    }
}
