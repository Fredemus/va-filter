#![feature(portable_simd)]
use core_simd::f32x4;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use va_filter::utils::AtomicOps;
use va_filter::{
    filter::LadderFilter, filter::SallenKey, filter::SallenKeyFast, filter_params_nih::FilterParams,
};
pub fn criterion_benchmark(c: &mut Criterion) {
    let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = Arc::new(FilterParams::new(should_update_filter.clone()));
    // let mut vst = VaFilter::default();
    let mut ladder = LadderFilter::new(params.clone());

    let mut sallen_key = SallenKey::new(params.clone());
    let mut sallen_key_fast = SallenKeyFast::new(params.clone());

    let val = -2. * params.g.get() as f64;

    c.bench_function("run sallenkey:", |b| {
        b.iter(|| black_box(sallen_key.homotopy_solver([0., val])))
    });

    c.bench_function("run sallenkey fast:", |b| {
        b.iter(|| black_box(sallen_key_fast.homotopy_solver([0., val])))
    });
    c.bench_function("run moog:", |b| {
        b.iter(|| black_box(ladder.run_filter_newton(f32x4::splat(0.0))))
    });
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
