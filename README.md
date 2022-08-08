# va-filter
Virtual analogue filters modelled with Topology Preserving Transforms and solved with Newton's method. implemented as a VST2 in Rust, using [Vizia](https://github.com/geom3trik/VIZIA) for the GUI.

## circuits
This plugin currently has 3 circuit models:

### Transistor ladder filter
This is a 4-pole lowpass ladder filter loosely based on the ones found in Moog synthesizers. It distorts nicely and is capable of stable self-oscillation when `k_ladder==4`, and can output other slopes too.

Resonance is limited by the BJT buffers. 

It converges very well, usually only taking 2 iterations, and almost never more than 4. Could just always do 2 especially when oversampled.

Circuit solved by applying KCL, finding the jacobian of the entire system and then applying newton's method.

### State-variable filter 
This is a 2-pole multimode filter loosely based on the one found in the Osc Oscar synhtesizer. It barely distorts the signal at all, only distorting the resonance because of some bad assumptions I had when making it. 

It's capable of outputting all basic filter modes (highpass, notch etc.) and self-oscillation.

OTA core, linear buffers.
Resonance is limited by a diode clipper on the damping feedback, boosting it when gain is high and it'd otherwise disappear because of the OTA nonlinearities.

Its convergence isn't great and will likely be removed or redone at some point when I find the time.

Circuit solved by applying KCL, finding the jacobian of the entire system and then applying newton's method.

### Sallen-key filter
This is a 2-pole lowpass filter loosely based on the one found in the second revision of the Korg MS20 synthesizer. It distorts really nicely and gets especially gnarly when resonance is high. My personal favorite. 

It's able to self-oscillate and starts doing so when its resonance is above 0.8. 

OTA core, nonlinear op-amp buffers.
Resonance is limited by a diode clipper, but it disappears quite quickly at high drives, look into tweaking diode constants.

Its convergence is generally good, but sometimes, especially when resonance and drive is very high, homotopy is needed to converge which gets slow. I'll look into speeding this up at some point. 
The parameter vector for the nonlinear contributions is just 2 entries long, meaning that it'd be very reasonable to create a lookup table to guarantee stable, fast runtime. 

Circuit solved by Holters & Zölzer's generalization of the DK-method. This method has a lot of advantages compared to the other approach, namely it's much better equipped for handling nonlinear voltage-controlled voltage sources such as op-amps and jacobian matrices are only necessary on a per-component basis, meaning it's not necessary to solve the whole system each iteration, speeding up iterations significantly.
Special thanks to Martin Holters and his amazing circuit emulation tool [ACME](https://github.com/HSU-ANT/ACME.jl) for the great work on circuit emulation and answering my questions when I got stuck.

Mono and missing SIMD optimization currently 

# Build Instructions

To run the standalone GUI:
```bash
cargo +nightly run --release --bin svf_gui_bin
```

To build the VST:
```bash
cargo +nightly build --release
```
## Packaging on OS X

On OS X VST plugins are packaged inside of loadable bundles. To package your VST as a loadable bundle you may use the osx_vst_bundler.sh script this library provides. 

Example: 
```bash
./osx_vst_bundler.sh Plugin target/release/plugin.dylib
Creates a Plugin.vst bundle
```