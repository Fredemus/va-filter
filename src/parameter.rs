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
pub struct Parameter<AtomicT: AtomicOps> {
    name: String,
    normalized_value: AtomicT,
    value: AtomicT,
    pub default: AtomicT::Item,
    pub min: AtomicT::Item,
    pub max: AtomicT::Item,
    display_func: fn(AtomicT::Item) -> String,
    // TODO: It might be very nice to have a set_func field like
    // set_func: fn(AtomicT::Item),
    // get_func: fn() -> AtomicT::Item,
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
    pub fn new(
        name: &str,
        default: AtomicT::Item,
        min: AtomicT::Item,
        max: AtomicT::Item,
        display_func: fn(AtomicT::Item) -> String,
        // set_func: fn(AtomicT::Item),
        // get_func: fn() -> AtomicT::Item,
    ) -> Parameter<AtomicT> {
        Parameter {
            name: String::from(name),
            normalized_value: AtomicT::new(default),
            value: AtomicT::new(default),
            default,
            min,
            max,
            display_func,
        }
    }

    pub fn get(&self) -> AtomicT::Item {
        self.value.get()
    }

    pub fn set(&self, x: AtomicT::Item) {
        self.value.set(x);
        self.normalized_value.set(self.from_range(x));
    }
    pub fn get_normalized(&self) -> AtomicT::Item {
        self.normalized_value.get()
    }

    pub fn set_normalized(&self, x: AtomicT::Item) {
        self.normalized_value.set(x);
        self.value.set(self.to_range(x));
    }
    pub fn get_display(&self) -> String {
        (self.display_func)(self.value.get())
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    // TODO: These don't work because addition etc. changes the type
    pub fn to_range(&self, x: AtomicT::Item) -> AtomicT::Item {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: AtomicT::Item) -> AtomicT::Item {
        (x - self.min) / (self.max - self.min)
    }
}
