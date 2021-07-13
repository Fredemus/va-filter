// use vst::util::AtomicFloat;
// use atomig::{Atomic, Ordering,Atom};
use crate::utils::*;
use std::ops::{Add, Div, Mul, Sub};
// pub fn to_range(bottom: f32, top: f32, x: f32) -> f32 {
//     x * (top - bottom) + bottom
// }

// pub fn from_range(bottom: f32, top: f32, x: f32) -> f32 {
//     (x - bottom) / (top - bottom)
// }

// TODO: Knobs don't work in the right way at all with set functions with a weird curve 
pub struct Parameter<AtomicT: AtomicOps> {
    name: String,
    normalized_value: AtomicT,
    value: AtomicT,
    pub default: AtomicT::Item,
    pub min: AtomicT::Item,
    pub max: AtomicT::Item,
    display_func: fn(AtomicT::Item) -> String,
    // TODO: It might be very nice to have a set_func field like
    pub set_func: fn(AtomicT::Item) -> AtomicT::Item,
    /// has to be the inverse of the set_func. Might make something to find it automatically
    pub get_func: fn(AtomicT::Item) -> AtomicT::Item,
}

impl<AtomicT: AtomicOps> Parameter<AtomicT>
// <AtomicT as utils::AtomicOps>::Item
where
    AtomicT::Item: Copy,
    AtomicT::Item: Sub<Output = AtomicT::Item>,
    AtomicT::Item: Add<Output = AtomicT::Item>,
    AtomicT::Item: Mul<Output = AtomicT::Item>,
    AtomicT::Item: Div<Output = AtomicT::Item>,
{
    /// this one assumes that you don't want a set function
    pub fn _new_no_setfunc(
        name: &str,
        default: AtomicT::Item,
        min: AtomicT::Item,
        max: AtomicT::Item,
        display_func: fn(AtomicT::Item) -> String,
    ) -> Parameter<AtomicT> {
        Parameter {
            name: String::from(name),
            normalized_value: AtomicT::new(default),
            value: AtomicT::new(default),
            default,
            min,
            max,
            display_func,
            set_func: |x| x,
            get_func: |x| x,
        }
    }
    pub fn new(
        name: &str,
        default: AtomicT::Item,
        min: AtomicT::Item,
        max: AtomicT::Item,
        display_func: fn(AtomicT::Item) -> String,
        set_func: fn(AtomicT::Item) -> AtomicT::Item,
        get_func: fn(AtomicT::Item) -> AtomicT::Item,
    ) -> Parameter<AtomicT> {
        Parameter {
            name: String::from(name),
            normalized_value: AtomicT::new(default),
            value: AtomicT::new(default),
            default,
            min,
            max,
            display_func,
            set_func,
            get_func,
        }
    }

    pub fn get(&self) -> AtomicT::Item {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.value.get()
    }
    /// assumes you have applied the set function elsewhere. Probably shouldn't be used?
    pub fn set(&self, x: AtomicT::Item) {
        // TODO: The knob fucks up because set_func needs to applied *before* it's to_range'd, which the knob does. 
        // current solution is dumb, but oh well
        self.set_normalized(self.from_range(x));
        // self.value.set((self.set_func)(x));
        // self.normalized_value
        //     .set(self.from_range((self.get_func)(x)));
        
        // let converted_to_useful = (self.set_func)(x);
        // self.value.set(converted_to_useful);
        // self.normalized_value.set(self.from_range(x));
    }
    pub fn get_normalized(&self) -> AtomicT::Item {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.normalized_value.get()
    }
    pub fn set_normalized(&self, x: AtomicT::Item) {
        // setting normalized_value with number between 0 and 1
        self.normalized_value.set(x);
        // setting value with the set_function
        self.value.set(self.to_range((self.set_func)(x)));
    }
    pub fn get_display(&self) -> String {
        (self.display_func)(self.value.get())
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    // TODO: Do we need/want these?
    pub fn to_range(&self, x: AtomicT::Item) -> AtomicT::Item {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: AtomicT::Item) -> AtomicT::Item {
        (x - self.min) / (self.max - self.min)
    }
}

// testing if param is set correctly
#[test]
fn test_cutoff_param() {
    let cutoff: Parameter<AtomicF32> = Parameter::new(
        "Cutoff",
        20000.,
        0.,
        20000.,
        |x| format!("{:.0} Hz", x),
        |x| (1.8f32.powf(10. * x - 10.)),
        |x: f32| 1. + 0.17012975 * (x).ln(),
    );

    println!("cutoff value: {}", cutoff.get());
    // println!("cutoff value2: {}", cutoff.get_display());
    println!("param value: {}", cutoff.get_normalized());
    cutoff.set_normalized(0.);
    println!("cutoff value: {}", cutoff.get());
    println!("param value: {}", cutoff.get_normalized());
    cutoff.set_normalized(0.1);
    println!("cutoff value: {}", cutoff.get());
    println!("param value: {}", cutoff.get_normalized());
    cutoff.set_normalized(0.5);
    println!("cutoff value: {}", cutoff.get());
    println!("param value: {}", cutoff.get_normalized());
    cutoff.set_normalized(1.0);
    println!("cutoff value: {}", cutoff.get());
    println!("param value: {}", cutoff.get_normalized());
}
