#![allow(dead_code)]
use std::sync::atomic::{self, Ordering};
// use std::ops::{Sub, Add, Mul, Div};
pub trait AtomicOps {
    type Item: Copy;
    fn new(v: Self::Item) -> Self;
    /// Loads a value from the atomic integer with relaxed ordering.
    fn get(&self) -> Self::Item;
    /// Stores a value into the atomic integer with relaxed ordering.
    fn set(&self, v: Self::Item);
}

/// Simple 32-bit floating point wrapper over `AtomicU32` with relaxed ordering.
pub struct AtomicF32(atomic::AtomicU32);
impl AtomicOps for AtomicF32 {
    type Item = f32;
    /// Create a new atomic 32-bit float with initial value `v`.
    fn new(v: f32) -> Self {
        AtomicF32(atomic::AtomicU32::new(v.to_bits()))
    }
    /// Loads a value from the atomic float with relaxed ordering.
    #[inline]
    fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    /// Stores a value into the atomic float with relaxed ordering.
    #[inline]
    fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}
/// Simple wrapper over `AtomicUsize` with relaxed ordering.
pub struct AtomicUsize(atomic::AtomicUsize);
impl AtomicOps for AtomicUsize {
    type Item = usize;
    /// Create a new atomic integer with initial value `v`.
    fn new(v: usize) -> AtomicUsize {
        AtomicUsize(atomic::AtomicUsize::new(v))
    }
    /// Loads a value from the atomic integer with relaxed ordering.
    #[inline]
    fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
    /// Stores a value into the atomic integer with relaxed ordering.
    #[inline]
    fn set(&self, v: usize) {
        self.0.store(v, Ordering::Relaxed)
    }
}
/// Simple wrapper over `AtomicBool` with relaxed ordering.
// pub struct AtomicBool(atomic::AtomicBool);
// #[allow(dead_code)]
// impl AtomicOps for AtomicBool {
//     type Item = bool;
//     /// Create a new atomic 8-bit integer with initial value `v`.
//     fn new(v: bool) -> AtomicBool {
//         AtomicBool(atomic::AtomicBool::new(v))
//     }

//     /// Loads a value from the atomic integer with relaxed ordering.
//     #[inline]
//     fn get(&self) -> bool {
//         self.0.load(Ordering::Relaxed)
//     }

//     /// Stores a value into the atomic integer with relaxed ordering.
//     #[inline]
//     fn set(&self, v: bool) {
//         self.0.store(v, Ordering::Relaxed)
//     }
// }
/// Simple wrapper over `AtomicI8` with relaxed ordering.
pub struct AtomicI8(atomic::AtomicI8);

impl AtomicI8 {
    /// Create a new atomic 8-bit integer with initial value `v`.
    pub fn new(v: i8) -> AtomicI8 {
        AtomicI8(atomic::AtomicI8::new(v))
    }

    /// Loads a value from the atomic integer with relaxed ordering.
    #[inline]
    pub fn get(&self) -> i8 {
        self.0.load(Ordering::Relaxed)
    }

    /// Stores a value into the atomic integer with relaxed ordering.
    #[inline]
    pub fn set(&self, v: i8) {
        self.0.store(v, Ordering::Relaxed)
    }
}
/// Simple 64-bit floating point wrapper over `AtomicU32` with relaxed ordering.
pub struct AtomicF64(atomic::AtomicU64);
#[allow(dead_code)]
impl AtomicF64 {
    /// Create a new atomic 32-bit float with initial value `v`.
    pub fn new(v: f64) -> Self {
        AtomicF64(atomic::AtomicU64::new(v.to_bits()))
    }

    /// Loads a value from the atomic float with relaxed ordering.
    #[inline]
    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    /// Stores a value into the atomic float with relaxed ordering.
    #[inline]
    pub fn set(&self, v: f64) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}
