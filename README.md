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
