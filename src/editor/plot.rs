// just a place to put the bode plot math 
use num::complex::Complex;

use std::f32::consts::PI;

pub fn lin_to_db(gain: f32) -> f32 {
        gain.log(10.0) * 20.0
}

// TODO: Low resonances don't get represented properly (should show a softer curve). 
// transfer function seems right in maple so probably something else
// It seems like the issue is the frequency axis (notch also changes size when cutoff changes)
pub fn get_svf_bode(cutoff: f32, k: f32, mode: usize) -> Vec<f32> {
    // bilinear transform, sample rate of 1
    // bogus sample rate, since the important part is just that the plot's max is 22050 Hz
    let g = (PI * cutoff / 44100.).tan();
    // resolution of bodeplot
    let len = 1000;

    let mut array = vec![Complex::new(1.,0.); len];
    let mut frequencies = vec![1.; len]; // frequency has to be in range [0, pi/2] for some reason?
    let base: f32 = 10.;
    for i in 0..len {
        // frequencies[i] = ((i + offset) as f32 / (len + offset) as f32).powi(2) * 2. * PI; 
        // frequencies[i] = (base.powf(i as f32 / (len -1) as f32) - 1.) / (base - 1.) * 2. * PI;
        // frequencies[i] = ((i) as f32 / (len) as f32).powi(10) * PI; 

        frequencies[i] = base.powf((i + 1) as f32 / (len as f32 ) * 3. - 3.)* PI / 2.;

    }
    // println!("frequencies: {:?}", frequencies.iter().map(|x| x * 44100. /( PI)).collect::<Vec<f32>>());
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
        3 => { // notch
            for i in 0..len {
                curr_s = frequencies[i] * j;
                array[i] = (g.powi(2) + curr_s.powi(2)) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
            }  
        }
        5 => { // bandpass (constant peak gain)
            for i in 0..len {
                curr_s = frequencies[i] * j;
                array[i] = (g * curr_s * k) / ((curr_s).powi(2) + k * g * curr_s + g.powi(2));
            }  
        }
        _ => (),
    }
    let mut amplitudes = vec![1.; len];
    for i in 0..len {
        amplitudes[i] = lin_to_db(array[i].norm());
        // amplitudes[i] = array[i].norm(); // for testing
    }

    amplitudes
}

#[test]
fn test_cutoff_value() {
    let amplitudes = get_svf_bode(25.14, 1./0.707, 1);
    // println!("{:?}", amplitudes.iter().max().unwrap());
    let len = 1000;

    let mut array = vec![Complex::new(1.,0.); len];
    let mut frequencies = vec![1.; len]; //? probably normalized angular frequency, that is from 0 to 2 pi
    // TODO: Frequency should be spaced not-linearly
    // offset to skip the stupid low frequencies. 15 means first value is 9.63 Hz
    let base: f32 = 10.;
    for i in 0..len {
        // frequencies[i] = base.powf(i as f32 / (len as f32 ) * 3. - 3.)  * 2. * PI;
        frequencies[i] = base.powf((i +1)as f32 / (len as f32 ) * 3. - 3.) * PI / 2.;
        // frequencies[i] = base.powf(i as f32 / (len as f32 ) * 3. - 3.) / PI;
        frequencies[i] *= 44100. / PI ;
    }
    println!("amps: {:?}", &amplitudes[0..20]);
    println!("freqs: {:?}", &frequencies[0..20]);

    // println!("50 hz: {}, {}", 0.00355754, 0.00355754 * 44100. / PI);
    // println!("20000 hz: {}", 20000. * PI / 44100.);
    // println!("22050 hz: {}", 22050. * PI / 44100.); // 1.5707963
    // println!("10 hz: {}", 10. * PI / 44100.); // 0.0007123793
    println!("current lowest: {}", frequencies[0] * 44100. / PI); 
    println!("current highest: {}", frequencies[999] * 44100. / PI); 
    
}
#[test]
fn test_frequencies() {
    let len = 1000;

    let mut array = vec![Complex::new(1.,0.); len];
    let mut frequencies = vec![1.; len]; //? probably normalized angular frequency, that is from 0 to 2 pi
    // TODO: Frequency should be spaced not-linearly
    // offset to skip the stupid low frequencies. 15 means first value is 9.63 Hz
    let base: f32 = 10.;
    for i in 0..len {
        // frequencies[i] = base.powf(i as f32 / (len as f32 ) * 3. - 3.)  * 2. * PI;
        frequencies[i] = base.powf(i as f32 / (len as f32 ) * 3. - 3.) ;
        // frequencies[i] = base.powf(i as f32 / (len as f32 ) * 3. - 3.) / PI;
        // frequencies[i] *= 44100. ;
    }
    println!("freqs: {:?}", &frequencies);

}

#[test]
fn test_normalized_values() {
    let mut amps = get_svf_bode(23.464352, 1./0.707, 0);

    // normalize like this, don't draw samples under 0?
    let maxmin = 20.;
    for x in &mut amps {
        *x = (*x - (-maxmin))/ (maxmin - (-maxmin))
    }


    println!("{:?}", amps);

}
#[test]
fn db_pls() {
    println!("{}", lin_to_db(4.));
}
#[test]
fn axis_math() {
    let len = 100;

    let mut frequencies = vec![1.; len]; //? probably normalized angular frequency, that is from 0 to pi (0 to nyquist)
    // TODO: Frequency should be spaced not-linearly
    // offset to skip the stupid low frequencies. 15 means first value is 9.63 Hz
    let offset = 0;
    // let base: f32 = 2.;
    let base: f32 = 10.;
    for i in 0..len {
        // frequencies[i-1] = ((i + offset) as f32).powi(10) ; 
        // frequencies[i] = 10f32.powf((i  / len) as f32); 
        frequencies[i] = (base.powf(i as f32 / (len -1) as f32) - 1.) / (base - 1.); 
        // frequencies[i] = 10f32.powi(i as i32); 
        // frequencies[i] = ((i) as f32 / (len) as f32).powi(10) * PI; 

        // frequencies[i] *= 22050.; 
    }
    // println!("frequencies: {:?}", frequencies.iter().map(|x| x * 44100. /( 2.* PI)).collect::<Vec<f32>>());
    println!("frequencies: {:?}", frequencies);

}

#[test]
fn axis_math2() {
    let len = 1000;

    let mut frequencies = vec![1.; len]; //? probably normalized angular frequency, that is from 0 to pi (0 to nyquist)
    // TODO: Frequency should be spaced not-linearly
    // offset to skip the stupid low frequencies. 15 means first value is 9.63 Hz
    let base: f32 = 10.;
    // let base: f32 = 2.;
    for i in 0..len {
        // frequencies[i] = (base.powf(i as f32 / (len) as f32 ) - 1.) / (base) ; 
        frequencies[i] = base.powf(i as f32 - (len - 1) as f32 );  // almost perfect, but what about offset?


        // TODO: We need a range from 0.001 to 1 as output. Should be -3 to 0 as input
        // let i_guy = 0.001 - 1. + i as f32 / (len as f32 );
        frequencies[i] = base.powf(i as f32 / (len as f32 ) * 3. - 3.) ;
        // frequencies[i] = i_guy;
        frequencies[i] *= 22050. * 2. * PI; 


    }
    // println!("frequencies: {:?}", frequencies.iter().map(|x| x * 44100. /( 2.* PI)).collect::<Vec<f32>>());
    println!("frequencies: {:?}", frequencies);
    println!("start frequency: {}", 10.  / 22050.);
    println!("start frequency actual: {}", 0.001 * 22050.);

}


