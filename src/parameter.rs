#![allow(dead_code)]
use crate::utils::*;
// draw_knob functions could be made simpler if parameter structs had the parameter_index saved
pub trait Parameter {
    fn get_normalized(&self) -> f32;
    fn get_normalized_default(&self) -> f32;

    fn set_normalized(&self, value: f32);
    fn get_display(&self) -> String;
    fn get_name(&self) -> String;
}
pub struct ParameterSmooth {
    pub name: String,
    normalized_value: AtomicF32,
    state: AtomicF32,
    // target: AtomicF32,   // normalized_value is target val
    value: AtomicF32,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    display_func: fn(f32) -> String,
    pub set_func: fn(f32) -> f32,
    /// has to be the inverse of the set_func. Might make something to find it automatically
    pub get_func: fn(f32) -> f32,
}
impl ParameterSmooth {
    pub fn new(
        name: &str,
        default: f32,
        min: f32,
        max: f32,
        display_func: fn(f32) -> String,
        set_func: fn(f32) -> f32,
        get_func: fn(f32) -> f32,
    ) -> ParameterSmooth {
        // TODO: Make an assert which fails if get and set doesn't match
        let a = ParameterSmooth {
            name: String::from(name),
            normalized_value: AtomicF32::new(default),
            state: AtomicF32::new(default),
            value: AtomicF32::new(default),
            default,
            min,
            max,
            display_func,
            set_func,
            get_func,
        };
        a.normalized_value.set((a.get_func)(a.from_range(default)));
        // a.state.set((a.get_func)(a.from_range(default)));
        a
    }
    #[inline]
    pub fn get(&self) -> f32 {
        self.value.get()
    }
    pub fn update(&self, filter_factor: f32) {
        // TODO: How to ensure that this is the unmodulated state?
        // TODO: Is it possible/better to make state non-normalized?
        // maybe normalized_value is target? And modulate uses state instead of normalized_value to set value?
        // let filter_factor = 0.01;
        self.state.set(
            self.state.get() + (self.normalized_value.get() - self.state.get()) * filter_factor,
        );
        self.value
            .set(self.to_range((self.set_func)(self.state.get())));
    }
    #[inline]
    /// this function allows modulation of the parameter without screwing with normalized_value
    /// should only be called once per sample or the previous modulation will get overwritten
    pub fn modulate(&self, mod_amt: f32) {
        self.value
            .set(self.to_range((self.set_func)((mod_amt + self.state.get()).clamp(0., 1.))));
    }

    #[inline]
    pub fn to_range(&self, x: f32) -> f32 {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }
}

impl Parameter for ParameterSmooth {
    #[inline]
    fn get_normalized(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.normalized_value.get()
    }
    fn get_normalized_default(&self) -> f32 {
        (self.get_func)(self.from_range(self.default))
    }
    #[inline]
    fn set_normalized(&self, x: f32) {
        // setting normalized_value with number between 0 and 1
        self.normalized_value.set(x);
        // setting value with the set_function
        // self.target.set(self.to_range((self.set_func)(x)));
    }
    fn get_display(&self) -> String {
        // (self.display_func)(self.value.get())
        (self.display_func)(self.to_range((self.set_func)(self.normalized_value.get())))
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

pub struct ParameterF32 {
    pub name: String,
    normalized_value: AtomicF32,
    value: AtomicF32,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    display_func: fn(f32) -> String,
    pub set_func: fn(f32) -> f32,
    /// has to be the inverse of the set_func. Might make something to find it automatically
    pub get_func: fn(f32) -> f32,
}

impl ParameterF32 {
    pub fn new(
        name: &str,
        default: f32,
        min: f32,
        max: f32,
        display_func: fn(f32) -> String,
        set_func: fn(f32) -> f32,
        get_func: fn(f32) -> f32,
    ) -> ParameterF32 {
        // TODO: Make an assert which fails if get and set doesn't match
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
    #[inline]
    pub fn get(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.value.get()
    }

    #[inline]
    /// this function allows modulation of the parameter without changing normalized_value
    /// should only be called once per sample or the previous modulation will get overwritten
    pub fn modulate(&self, mod_amt: f32) {
        self.value.set(self.to_range((self.set_func)(
            (mod_amt + self.normalized_value.get()).clamp(0., 1.),
        )));
    }

    #[inline]
    pub fn to_range(&self, x: f32) -> f32 {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }
}
impl Parameter for ParameterF32 {
    #[inline]
    fn get_normalized(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.normalized_value.get()
    }
    fn get_normalized_default(&self) -> f32 {
        (self.get_func)(self.from_range(self.default))
    }
    #[inline]
    fn set_normalized(&self, x: f32) {
        // setting normalized_value with number between 0 and 1
        self.normalized_value.set(x);
        // setting value with the set_function
        self.value.set(self.to_range((self.set_func)(x)));
    }
    fn get_display(&self) -> String {
        (self.display_func)(self.value.get())
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }
}
pub struct ParameterUsize {
    name: String,
    normalized_value: AtomicF32,
    value: AtomicUsize,
    pub default: usize,
    pub min: f32,
    pub max: f32,
    pub display_func: fn(usize) -> String,
}

impl ParameterUsize {
    pub fn new(
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
        };
        a.normalized_value.set(a.from_range(default as f32));
        a
    }
    #[inline]
    pub fn get(&self) -> usize {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.value.get()
    }

    #[inline]
    /// this function allows modulation of the parameter without screwing with normalized_value
    /// should only be called once per sample or the previous modulation will get overwritten
    pub fn modulate(&self, mod_amt: f32) {
        self.value
            .set(self.to_range((mod_amt + self.normalized_value.get()).clamp(0., 1.)) as usize);
    }
    #[inline]
    pub fn to_range(&self, x: f32) -> f32 {
        x * (self.max - self.min) + self.min
    }

    pub fn from_range(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }
}
impl Parameter for ParameterUsize {
    #[inline]
    fn get_normalized(&self) -> f32 {
        // let converted_to_normal = (self.get_func)(self.value.get());
        // converted_to_normal
        self.normalized_value.get()
    }
    fn get_normalized_default(&self) -> f32 {
        self.from_range(self.default as f32)
    }
    #[inline]
    fn set_normalized(&self, x: f32) {
        // setting normalized_value with number between 0 and 1
        self.normalized_value.set(x);
        // setting value with the set_function
        self.value.set(self.to_range(x) as usize);
    }
    fn get_display(&self) -> String {
        (self.display_func)(self.value.get())
    }

    fn get_name(&self) -> String {
        self.name.clone()
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
