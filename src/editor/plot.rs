// just a place to put the bode plot math 
use num::complex::Complex;

use std::f32::consts::PI;

pub fn lin_to_db(gain: f32) -> f32 {
        gain.log(10.0) * 20.0
}

// TODO: Low resonances don't get represented properly (should show a softer curve). 
// transfer function seems right in maple so probably something else
pub fn get_svf_bode(cutoff: f32, k: f32, mode: usize, nonlinear: bool) -> Vec<f32> {
    // bilinear transform, sample rate of 1
    // bogus sample rate, since the important part is just that the plot's max is 22050 Hz
    let g = (PI * cutoff / 44100.).tan();

    // resolution of bodeplot
    let len = 1000;
    // the resonance is 2 times lower in nonlinear mode i think? Verify!
    // if nonlinear {
    //     k = k * 2.;
    // }
    let mut array = vec![Complex::new(1.,0.); len];
    let mut frequencies = vec![1.; len]; //? probably normalized angular frequency, that is from 0 to pi (0 to nyquist)
    // TODO: Frequency should be spaced not-linearly
    // offset of 500 to skip the stupid low frequencies. Potentially use sample rate to do this better?
    let offset = 750;
    for i in 0..len {
        frequencies[i] = ((i + offset) as f32 / (len + offset) as f32).powi(10) * 2. * PI; 
        // frequencies[i] = ((i) as f32 / (len) as f32).powi(10) * PI; 
    }
    // println!("frequencies: {:?}", frequencies.iter().map(|x| x * 44100. /( 2.* PI)).collect::<Vec<f32>>());
    let j = Complex::new(0., 1.);
    let mut curr_s : Complex::<f32>;
    match mode {
        0 => { // lowpass
            for i in 0..len {
                curr_s = frequencies[i] * j;
                array[i] = g.powi(2) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
            }
        }
        1 => { // highpass
            for i in 0..len {
                curr_s = frequencies[i] * j;
                array[i] = curr_s.powi(2) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
            }
        }
        2 => { // bandpass
            for i in 0..len {
                curr_s = frequencies[i] * j;
                array[i] = (g * curr_s) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
            }    
        }
        _ => (),
    }
    let mut amplitudes = vec![1.; len];
    for i in 0..len {
        amplitudes[i] = lin_to_db(array[i].norm());
        // amplitudes[i] = array[i].norm(); // for testing
    }
    // for x in &mut array {
    //     x = (*x).norm()
    // }
    // something to draw the absolute plot
    // println!("{:?}", amplitudes);

    return amplitudes;

    
}

#[test]
fn test_values() {
    println!("bogfrji");
    let amplitudes = get_svf_bode(1000., 1./20., 0, true);
    println!("{:?}", amplitudes);
    // println!("{:?}", amplitudes.iter().max().unwrap());

}

#[test]
fn test_normalized_values() {
    let mut amps = get_svf_bode(1000., 1./20., 0, true);
    // amplitudes = amplitudes.iter().map(|x| x.max(-12.)).collect();

    // normalize like this, don't draw samples under 0?
    let maxmin = 20.;
    for x in &mut amps {
        *x = (*x - (-maxmin))/ (maxmin - (-maxmin))
    }


    println!("{:?}", amps);
    println!("{}", (0. - (-maxmin))/ (maxmin - (-maxmin)));

}
#[test]
fn db_pls() {
    println!("{}", lin_to_db(4.));
}


