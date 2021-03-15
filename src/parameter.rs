// use vst::util::AtomicFloat;
// use atomig::{Atomic, Ordering,Atom};
use crate::utils::*;
pub fn to_range(bottom: f32, top: f32, x: f32) -> f32 {
    x * (top - bottom) + bottom
}

pub fn from_range(bottom: f32, top: f32, x: f32) -> f32 {
    (x - bottom) / (top - bottom)
}
pub struct Parameter<AtomicT: Default + AtomicOps> {
    name: String,
    normalized_value: AtomicT,
    value: AtomicT,
    pub default: AtomicT::Item,
    pub min: AtomicT::Item,
    pub max: AtomicT::Item,
    display_func: fn(AtomicT::Item) -> String,
}

impl<AtomicT: Default + AtomicOps> Parameter<AtomicT> 
// <AtomicT as utils::AtomicOps>::Item
// where
//     T: AtomicOps::Item
//     <T as AtomicOps>::Item: T
{
    pub fn new(
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
        }
    }

    // pub fn get_normalized(&self) -> T {
    //     self.normalized_value.get()
    // }

    // pub fn set_normalized(&self, x: T) {
    //     self.normalized_value.set(x);
    //     self.value.set(to_range(self.min, self.max, x));
    // }

    pub fn get(&self) -> AtomicT::Item {
        self.value.get()
    }

    pub fn set(&self, x: AtomicT::Item) {
        self.value.set(x);
        // self.normalized_value.set(from_range(self.min, self.max, x));
    }

    pub fn get_display(&self) -> String {
        (self.display_func)(self.value.get())
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }
}
