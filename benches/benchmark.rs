#![feature(portable_simd)]
#![allow(dead_code)]
use core_simd::simd::f32x4;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use va_filter::filter::svf::SvfCoreFast;
// use va_filter::utils::AtomicOps;
use va_filter::{
    filter::sallen_key::SallenKeyCore, filter::sallen_key::SallenKeyCoreFast, filter::svf::SvfCore,
    filter::LadderFilter, filter_params::FilterParams,
};
pub fn criterion_benchmark(c: &mut Criterion) {
    let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = Arc::new(FilterParams::new(should_update_filter.clone()));

    let mut ladder = LadderFilter::new(params.clone());

    let mut _sallen_key = SallenKeyCore::new(params.clone());
    let mut _sallen_key_fast = SallenKeyCoreFast::new(params.clone());
    let mut _svf = SvfCore::new(params.clone());
    let mut _svf_fast = SvfCoreFast::new(params.clone());

    c.bench_function("run svf:", |b| {
        b.iter(|| {
            black_box({
                _svf.tick_dk(1.);
                _svf.reset()
            })
        })
    });

    c.bench_function("run svf fast:", |b| {
        b.iter(|| {
            black_box({
                _svf_fast.tick_dk(1.);
                _svf_fast.reset()
            })
        })
    });

    c.bench_function("run sallenkey fast:", |b| {
        b.iter(|| {
            black_box({
                _sallen_key_fast.reset();
                _sallen_key_fast.tick_dk(1.)
            })
        })
    });

    c.bench_function("run sallenkey:", |b| {
        b.iter(|| {
            black_box({
                _sallen_key.tick_dk(1.);
                _sallen_key.reset()
            })
        })
    });

    c.bench_function("run moog:", |b| {
        b.iter(|| black_box(ladder.run_filter_newton(f32x4::splat(0.0))))
    });
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
