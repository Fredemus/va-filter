// just a place to put the bode plot math
use num::complex::Complex;

use std::f32::consts::PI;

pub fn lin_to_db(gain: f32) -> f32 {
    gain.log10() * 20.0
}

/// really cheap tan, works well enough in range [0, 0.5 * pi], but gets a bit inaccurate close to the upper bound
pub fn _cheap_tan(x: f32) -> f32 {
    (-0.66666667 * x.powi(3) + x) / (1. - 0.4 * x.powi(2))
}

fn get_filter_bode(
    cutoff: f32,
    k: f32,
    mode: usize,
    filter_type: usize,
    len: usize,
) -> Vec<Complex<f32>> {
    let g = cutoff;
    let mut frequencies = vec![1.; len]; 
    let mut array = vec![Complex::new(1., 0.); len];
    // frequency map setup
    let min: f32 = 20.;
    let max: f32 = 20000.;
    let minl = min.log2();
    let range = max.log2() - minl;
    
    for i in 0..len {
        frequencies[i] = 2.0f32.powf(((i as f32 / len as f32) * range) + minl);
    }
    let j = Complex::new(0., 1.);
    let mut curr_s: Complex<f32>;
    match filter_type {
        // state variable filter
        0 => {
            // let k = res.powf(0.2) * (0.05 - 10.) + 10.;
            match mode {
                0 => {
                    // lowpass
                    for i in 0..len {
                        curr_s = frequencies[i] * j;
                        array[i] = g.powi(2) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
                    }
                }
                1 => {
                    // highpass
                    for i in 0..len {
                        curr_s = frequencies[i] * j;
                        array[i] = curr_s.powi(2) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
                    }
                }
                2 => {
                    // bandpass
                    for i in 0..len {
                        curr_s = frequencies[i] * j;
                        array[i] = (g * curr_s) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
                    }
                }
                3 => {
                    // notch
                    for i in 0..len {
                        curr_s = frequencies[i] * j;
                        array[i] = (g.powi(2) + curr_s.powi(2))
                            / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
                    }
                }
                4 => {
                    // bandpass (constant peak gain)
                    for i in 0..len {
                        curr_s = frequencies[i] * j;
                        array[i] =
                            (g * curr_s * k) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
                    }
                }
                _ => (),
            }
        }
        // transistor ladder filter
        1 => {
            for i in 0..len {
                curr_s = frequencies[i] * j;
                // could potentially be optimized, i think
                array[i] = ((1. + k) * (1. + curr_s / g).powi(3 - mode as i32))/ (k + (1. + curr_s / g).powi(4));
                // array[i] =
                //     ((1. + curr_s / g).powi(3 - mode as i32)) / (k + (1. + curr_s / g).powi(4));
            }
        }
        _ => (),
    }
    return array;
}

pub fn get_amplitude_response(
    cutoff: f32,
    k: f32,
    mode: usize,
    filter_type: usize,
    len: usize,
) -> Vec<f32> {
    let array = get_filter_bode(cutoff, k, mode, filter_type, len);
    let mut amplitudes = vec![1.; len];
    for i in 0..len {
        amplitudes[i] = lin_to_db(array[i].norm());
    }
    // make notch draw a lil nicer at high q-factors (the problem is that there might not be a freq sample at the cutoff)
    if filter_type == 0 && mode == 3 {
        let min = amplitudes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in the filter response"))
            .unwrap()
            .0;
        amplitudes[min] = -200.;
    }
    // round max reso value to the correct, for same reason as above
    else if filter_type == 0 && mode != 4 && k < 0.5 {
        let max = amplitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in the filter response"))
            .unwrap()
            .0;
        amplitudes[max] = lin_to_db(1.0 / k);
    }
    // TODO: I'd like to do this for the ladder filter too, but I couldn't find a formula for its resonance
    amplitudes
}

pub fn get_phase_response(
    cutoff: f32,
    k: f32,
    mode: usize,
    filter_type: usize,
    len: usize,
) -> Vec<f32> {
    let array = get_filter_bode(cutoff, k, mode, filter_type, len);
    let mut phases = vec![1.; len];
    for i in 0..len {
        phases[i] = array[i].arg();
    }
    // make notch and 4-pole ladder draw a lil nicer at high q-factors 
    // (the problem is that there might not be a freq sample at the cutoff)
    // if mode == 3 && (filter_type == 0  || filter_type == 1) {
    if mode == 3 {
        let min = phases
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in the filter response"))
            .unwrap()
            .0;
        let max = phases
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in the filter response"))
            .unwrap()
            .0;
        if phases[min] < -1. {
            phases[min] = -PI;
        }
        if phases[max] > 1. {
            phases[max] = PI;
        }
    }
    phases
}

#[test]
fn test_cutoff_value() {
    let len = 1000;
    let amplitudes = get_amplitude_response(25.1425 * 2., 1. / 0.707, 0, 0, len);
    // println!("{:?}", amplitudes.iter().max().unwrap());

    let mut frequencies = vec![1.; len];
    let base: f32 = 10.;
    for i in 0..len {
        frequencies[i] = base.powf((i + 1) as f32 / (len as f32) * 3. - 3.) * PI / 2.;
        // turns the frequency to hertz
        frequencies[i] *= 44100. / PI;
    }
    let max_amp = 50;
    println!("amps: {:?}", &amplitudes[0..max_amp]);
    println!("freqs: {:?}", &frequencies[0..max_amp]);

    println!("current lowest: {}", frequencies[0]);
    println!("current highest: {}", frequencies[999]);
}
#[test]
fn test_ladder_value() {
    let len = 1000;
    let amplitudes = get_amplitude_response(25.1425, 3.99, 3, 1, len);
    // println!("{:?}", amplitudes.iter().max().unwrap());

    let mut frequencies = vec![1.; len];
    let base: f32 = 10.;
    for i in 0..len {
        frequencies[i] = base.powf((i + 1) as f32 / (len as f32) * 3. - 3.) * PI / 2.;
        // turns the frequency to hertz
        frequencies[i] *= 44100. / PI;
    }
    let min_idx = 50;
    let max_idx = 100;
    println!("amps: {:?}", &amplitudes[min_idx..max_idx]);
    println!("freqs: {:?}", &frequencies[min_idx..max_idx]);

    println!(
        "highest amp: {}",
        amplitudes.into_iter().reduce(f32::max).unwrap()
    );
    println!("current lowest: {}", frequencies[0]);
    println!("current highest: {}", frequencies[999]);
}
#[test]
fn db_print() {
    println!("{}", lin_to_db(0.1));
}
#[test]
fn pr_print() {
    println!("{:?}", get_phase_response(10000., 0.707, 1, 1, 1000));
}
