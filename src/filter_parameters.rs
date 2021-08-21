use super::parameter::{ParameterF32, ParameterUsize};
use super::utils::*;
use std::f32::consts::PI;

pub struct FilterParameters {
    // the "cutoff" parameter. Determines how heavy filtering is
    // cutoff: AtomicFloat,
    // cutoff: Parameter<AtomicF32>,
    pub g: AtomicF32,
    pub sample_rate: AtomicF32,

    pub cutoff: ParameterF32,
    pub res: ParameterF32,
    pub zeta: AtomicF32,
    pub k_ladder: AtomicF32,

    pub drive: ParameterF32,
    pub mode: ParameterUsize,
    pub slope: ParameterUsize,
    pub filter_type: AtomicUsize,
    // cutoff: Params,
    // res: Params,
    // drive: Params,
    // mode: Params,
}
impl FilterParameters {
    // transform resonance parameter into something more useful for the filter
    pub fn set_resonances(&self) {
        let res = self.res.get_normalized();
        self.zeta.set(5. - 4.9 * res);
        self.k_ladder.set(res.powi(2) * 3.8 - 0.2);
    }
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
//             // What's the best way to handle the reference here? removed for now i guess
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
    // todo: How do we make sure g gets set? Maybe bake g into cutoff and have display func show cutoff
    fn default() -> FilterParameters {
        let a = FilterParameters {
            sample_rate: AtomicF32::new(48000.),
            filter_type: AtomicUsize::new(0),
            cutoff: ParameterF32::new(
                "Cutoff",
                10000.,
                0.,
                20000.,
                |x| format!("{:.0} Hz", x),
                |x| (1.8f32.powf(10. * x - 10.)),
                |x: f32| 1. + 0.17012975 * (x).ln(),
            ),
            g: AtomicF32::new((PI * 10000. / 48000.).tan()),
            // TODO: Res fucks up at low values, caused by the formula being dumb
            // Maybe just rewrite filter equations to divide by res so we can set res as q-factor directly?
            // should be way easier to work with
            // res: (ParameterF32::new(
            //     "Resonance",
            //     1. / 0.707,
            //     10.,
            //     0.05,
            //     |x| format!("{:.2}", 1. / x),
            //     |x: f32| x.powf(0.2),
            //     |x: f32| x.powi(5),
            //     // |x| 2f32.powf(-11. * x),
            //     // |x: f32| (x).ln() * -0.13115409,
            // )),
            res: (ParameterF32::new(
                "Resonance",
                0.5,
                0.,
                1.,
                |x| format!("{:.2} %", x * 100.),
                |x: f32| x,
                |x: f32| x,
                // |x| 2f32.powf(-11. * x),
                // |x: f32| (x).ln() * -0.13115409,
            )),
            drive: (ParameterF32::new(
                "Drive",
                0.,
                0.,
                14.8490,
                |x: f32| format!("{:.2} dB", 20. * (x + 1.).log10()),
                |x| x.powi(2),
                |x| x.sqrt(),
            )),
            mode: (ParameterUsize::new(
                "Filter mode",
                0,
                0,
                4,
                |x| match x {
                    0 => format!("Lowpass"),
                    1 => format!("Highpass"),
                    2 => format!("Bandpass 1"),
                    3 => format!("Notch"),
                    _ => format!("Bandpass 2"),
                },
                |x| x,
                |x| x,
            )),
            slope: (ParameterUsize::new(
                "Filter slope",
                3,
                0,
                3,
                |x| match x {
                    0 => format!("Lp6"),
                    1 => format!("LP12"),
                    2 => format!("LP18"),
                    3 => format!("LP24"),
                    _ => format!("???"),
                },
                |x| x,
                |x| x,
            )),
            k_ladder: AtomicF32::new(0.),
            zeta: AtomicF32::new(0.),
        };
        a.g.set((PI * a.cutoff.get() / (a.sample_rate.get())).tan());
        a.set_resonances();
        a
    }
}

#[test]
fn test_res_param() {
    let params = FilterParameters::default();
    let res = params.res;

    res.set_normalized(0.0);
    println!("res value: {}", res.get());
    println!("display value: {}", res.get_display());
    println!("param value: {}", res.get_normalized());
    res.set_normalized(0.1);
    println!("res value: {}", res.get());
    println!("display value: {}", res.get_display());
    println!("param value: {}", res.get_normalized());
    res.set_normalized(0.5);
    println!("res value: {}", res.get());
    println!("display value: {}", res.get_display());
    println!("param value: {}", res.get_normalized());
    res.set_normalized(1.);
    println!("res value: {}", res.get());
    println!("display value: {}", res.get_display());
    println!("param value: {}", res.get_normalized());
}
