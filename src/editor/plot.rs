// just a place to put the bode plot math 
use num::complex::Complex;

use std::f32::consts::PI;

pub fn lin_to_db(gain: f32) -> f32 {
        gain.log(10.0) * 20.0
}


pub fn get_bode_array(g: f32, k: f32, mode: usize ) -> Vec<f32> {
    // resolution of bodeplot
    let len = 1000;

    let mut array = vec![Complex::new(1.,0.); len];
    let mut frequencies = vec![1.; len]; //? probably normalized angular frequency, that is from 0 to pi
    // TODO: Frequency should be spaced not-linearly
    // offset of 500 to skip the stupid low frequencies. Potentially use sample rate to do this better?
    for i in 0..len {
        frequencies[i] = ((i + 500) as f32 / (len + 500) as f32).powi(10) * PI ; 
    }
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
    }
    // for x in &mut array {
    //     x = (*x).norm()
    // }
    // something to draw the absolute plot
    // println!("{:?}", amplitudes);

    return amplitudes;

    
}

#[test]
fn test_plot() {
    println!("bogfrji");
    let amplitudes = get_bode_array(0.5, 1./20., 1);
    println!("{:?}", amplitudes);

}




