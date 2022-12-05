use core_simd::simd::f32x4;

// basic DC-filter from Understanding Digital Signal Processing by Richard Lyons
pub struct DcFilter {
    y0: f32x4,
    x0: f32x4,
    // higher alpha moves the cutoff lower, but also makes it settle slower. See if it's reasonable to make it higher
    alpha: f32x4,
}

impl Default for DcFilter {
    fn default() -> Self {
        Self {
            y0: f32x4::splat(0.),
            x0: f32x4::splat(0.),
            // alpha: f32x4::splat(0.975),
            alpha: f32x4::splat(0.9999),
        }
    }
}

impl DcFilter {
    pub fn process(&mut self, input: f32x4) -> f32x4 {
        let y_new = input - self.x0 + self.alpha * self.y0;
        self.x0 = input;
        self.y0 = y_new;

        y_new
    }
}
