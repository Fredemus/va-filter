// use vst::util::AtomicFloat;
// use atomig::{Atomic, Ordering,Atom};
use crate::utils::*;
// use std::ops::{Add, Div, Mul, Sub};
// pub fn to_range(bottom: f32, top: f32, x: f32) -> f32 {
//     x * (top - bottom) + bottom
// }

// pub fn from_range(bottom: f32, top: f32, x: f32) -> f32 {
//     (x - bottom) / (top - bottom)
// }

pub struct ParameterF32 {
    name: String,
    normalized_value: AtomicF32,
    value: AtomicF32,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    display_func: fn(f32) -> String,
    // TODO: It might be very nice to have a set_func field like
    pub set_func: fn(f32) -> f32,
    /// has to be the inverse of the set_func. Might make something to find it automatically
    pub get_func: fn(f32) -> f32,
}

impl ParameterF32
// <AtomicT as utils::AtomicOps>::Item
{
    /// this one assumes that you don't want a set function
    pub fn _new_no_setfunc(
        name: &str,
        default: f32,
        min: f32,
        max: f32,
        display_func: fn(f32) -> String,
    ) -> ParameterF32 {
        let a = ParameterF32 {
            name: String::from(name),
            normalized_value: AtomicF32::new(default),
            value: AtomicF32::new(default),
            default,
            min,
            max,
            display_func,
            set_func: |x| x,
            get_func: |x| x,
        };
        a.normalized_value.set((a.get_func)(a.from_range(default)));
        a
    }
    pub fn new(
        name: &str,
        default: f32,
        min: f32,
        max: f32,
        display_func: fn(f32) -> String,
        set_func: fn(f32) -> f32,
        get_func: fn(f32) -> f32,
    ) -> ParameterF32 {
        let a = ParameterF32 {
            name: String::from(name),
            normalized_value: AtomicF32::new(default),
            value: AtomicF32::new(default),
            default,
            min,
            max,
            display_func,
            set_func,
            get_func,
        };
        a.normalized_value.set((a.get_func)(a.from_range(default)));
        a
    }

    pub fn get(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.value.get()
    }
    pub fn get_normalized(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.normalized_value.get()
    }
    pub fn set_normalized(&self, x: f32) {
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
    pub fn to_range(&self, x: f32) -> f32 {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }
}
pub struct ParameterUsize {
    name: String,
    normalized_value: AtomicF32,
    value: AtomicUsize,
    pub default: usize,
    pub min: f32,
    pub max: f32,
    display_func: fn(usize) -> String,
    // TODO: Usize maybe doesn't need a set and get_func
    pub set_func: fn(f32) -> f32,
    /// has to be the inverse of the set_func. Might make something to find it automatically
    pub get_func: fn(f32) -> f32,
}

impl ParameterUsize
// <AtomicT as utils::AtomicOps>::Item
{
    /// this one assumes that you don't want a set function
    pub fn _new_no_setfunc(
        name: &str,
        default: usize,
        min: usize,
        max: usize,
        display_func: fn(usize) -> String,
    ) -> ParameterUsize {
        let a = ParameterUsize {
            name: String::from(name),
            normalized_value: AtomicF32::new(0.),
            value: AtomicUsize::new(default),
            default,
            min: min as f32,
            max: max as f32,
            display_func,
            set_func: |x| x,
            get_func: |x| x,
        };
        a.normalized_value.set((a.get_func)(a.from_range(default as f32)));
        a
    }
    pub fn new(
        name: &str,
        default: usize,
        min: usize,
        max: usize,
        display_func: fn(usize) -> String,
        set_func: fn(f32) -> f32,
        get_func: fn(f32) -> f32,
    ) -> ParameterUsize {
        let a = ParameterUsize {
            name: String::from(name),
            normalized_value: AtomicF32::new(0.),
            value: AtomicUsize::new(default),
            default,
            min: min as f32,
            max: max as f32,
            display_func,
            set_func,
            get_func,
        };
        a.normalized_value.set((a.get_func)(a.from_range(default as f32)));
        a
    }

    pub fn get(&self) -> usize {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.value.get()
    }
    pub fn get_normalized(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.normalized_value.get()
    }
    pub fn set_normalized(&self, x: f32) {
        // setting normalized_value with number between 0 and 1
        self.normalized_value.set(x);
        // setting value with the set_function
        self.value.set(self.to_range((self.set_func)(x)) as usize);
    }
    pub fn get_display(&self) -> String {
        (self.display_func)(self.value.get())
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    // TODO: Do we need/want these?
    pub fn to_range(&self, x: f32) -> f32 {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }
}
// testing if param is set correctly
#[test]
fn test_cutoff_param() {
    let cutoff: ParameterF32 = ParameterF32::new(
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
