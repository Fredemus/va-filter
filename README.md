# svf_filter
A nonlinear model of a state variable filter implemented as a VST in Rust


# TO DO

* Parameter<usize> needs to use f32 for its normalized_value. Restate Parameter<T>::normalized_value to always be f32
* the variable g should be "baked into" parameter "cutoff".

* get cutoff and resonance parameters to work. (implement set_/get_normalized). To do this we need to use set_normalized() and  get_normalized() on cutoff and resonance under PluginParameters trait. To keep the parameters scaled nicely like they currently are, we will need a get-func and set-func on the parameter-struct - could be something like changing the value logarithmically. Bring the implementation into the parameter-struct.

* Steppy knob for adjusting the filter-type.

* The quality of the filter can be improved by adding iterative calculations for the non-linear feedback.
Potentially runge-kutta method? https://github.com/ddiakopoulos/MoogLadders/blob/master/src/RKSimulationModel.h 