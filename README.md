# va-filter
Virtual analogue filters modelled with Topology Preserving Transforms and solved with Newton's method. implemented as a VST2 in Rust, using [Vizia](https://github.com/geom3trik/VIZIA) for the GUI.

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