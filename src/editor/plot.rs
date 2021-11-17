// just a place to put the bode plot math
// TODO: ladder and svf doesn't agree on where cutoff is. Which one is wrong? probably ladder but not sure
use num::complex::Complex;

use std::f32::consts::PI;

pub fn lin_to_db(gain: f32) -> f32 {
    gain.log10() * 20.0
}

/// really cheap tan, works well enough in range [0, 0.5 * pi], but gets a bit inaccurate close to the upper bound
pub fn _cheap_tan(x: f32) -> f32 {
    (-0.66666667 * x.powi(3) + x) / (1. - 0.4 * x.powi(2))
}

pub fn get_filter_bode(cutoff: f32, k: f32, mode: usize, filter_type: usize) -> Vec<f32> {
    // bilinear transform
    // bogus sample rate of 44100, since it just changes the plot's max value and 22050 seems reasonable
    let g = (PI * cutoff / 44100.).tan();
    // resolution of bodeplot
    let len = 500;

    let mut array = vec![Complex::new(1., 0.); len];
    let mut frequencies = vec![1.; len]; // frequency has to be in range [0, pi/2] because that's the range of g from the BLT
    let base: f32 = 10.;
    for i in 0..len {
        frequencies[i] = base.powf((i + 1) as f32 / (len as f32) * 3. - 3.) * PI / 2.;
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
                array[i] = ((1. + k) * (1. + curr_s / g).powi(3 - mode as i32))
                    / (k + (1. + curr_s / g).powi(4));
            }
        }
        _ => (),
    }

    let mut amplitudes = vec![1.; len];
    for i in 0..len {
        amplitudes[i] = lin_to_db(array[i].norm());
    }
    amplitudes
}

#[test]
fn test_cutoff_value() {
    let amplitudes = get_filter_bode(25.1425 * 2., 1. / 0.707, 3, 0);
    // println!("{:?}", amplitudes.iter().max().unwrap());
    let len = 1000;

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
    let amplitudes = get_filter_bode(25.1425, 3.99, 3, 1);
    // println!("{:?}", amplitudes.iter().max().unwrap());
    let len = 1000;

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
