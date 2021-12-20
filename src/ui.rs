use std::sync::Arc;
use crate::utils::*;
use vizia::*;
use vst::host::Host;
use vst::plugin::HostCallback;
use vst::plugin::PluginParameters;
use crate::editor::EditorState;
use crate::parameter::*;
use crate::FilterParameters;

const STYLE: &str = r#"
    label {
        font-size: 20;
        color: #C2C2C2;
    }
    knob {
        width: 70px;
        height: 70px;
    }
    
    knob .track {
        background-color: #ffb74d;
    }
"#;

#[derive(Lens)]
pub struct Params {
    params: Arc<FilterParameters>,
    host: Option<HostCallback>,
}

#[derive(Debug)]
pub enum ParamChangeEvent {
    _SetGain(f32),
    AllParams(i32, f32),
}

impl Model for Params {
    fn event(&mut self, _cx: &mut Context, event: &mut Event) {
        if let Some(param_change_event) = event.message.downcast() {
            match param_change_event {
                ParamChangeEvent::_SetGain(_new_gain) => {
                    
                }
                ParamChangeEvent::AllParams(parameter_index, new_value,) => {
                    // host needs to know that the parameter should/has changed
                    if let Some(host) = self.host {
                        host.begin_edit(*parameter_index);
                        host.automate(*parameter_index, *new_value);
                        host.end_edit(*parameter_index);
                    }
                    // set_parameter is on the PluginParameters trait
                    else {
                        self.params.set_parameter(*parameter_index, *new_value);
                    }
                }
            }
        }
    }
}

pub fn plugin_gui(cx: &mut Context, state: Arc<EditorState> ) {
    cx.add_theme(STYLE);

    Params {
        params: state.params.clone(),
        host: state.host,
    }.build(cx);
    //TODO: Just make a method on params getting these, have an event that gets em maybe?
    // let cutoff_default = state.params.cutoff.get_normalized_default();
    // let res_default = state.params.res.get_normalized_default();
    // let drive_default = state.params.drive.get_normalized_default();
    // let mode_default = state.params.mode.get_normalized_default();

    // TODO: How to move Hstack up/down
    HStack::new(cx, |cx| {
        VStack::new(cx, |cx|{
            Label::new(cx, "Cutoff");
            // let param_ref = &params;
            let map = GenericMap::new(0.0, 1.0, ValueScaling::Linear, DisplayDecimals::Two, None);
            
            // Knob::new(cx, map.clone(), params.osc_p[0].volume.get_normalized_default()).on_changing(cx, |knob, cx|{
                Knob::new(cx, map.clone(), 0.5).on_changing(cx, |knob, cx,|{
    
                // cx.emit(ParamChangeEvent::SetGain(knob.normalized_value));
                cx.emit(ParamChangeEvent::AllParams(0, knob.normalized_value))
            });
            Binding::new(cx, Params::params, move |cx, params|{
                // let amplitude = params.get(cx).osc_p[0].volume.get();
                // let amplitude = params.get(cx).osc_p[0].volume.get_display();
                // Label::new(cx, &map.normalized_to_display(amplitude));
                Label::new(cx, &params.get(cx).cutoff.get_display());
                // Label::new(cx, &params.get(cx).osc_p[0].volume.get_display());
    
            });
            
    
    
        }).child_space(Stretch(1.0)).row_between(Pixels(10.0));
    
        VStack::new(cx, |cx|{
            Label::new(cx, "Res");
            let map = GenericMap::new(0.0, 1.0, ValueScaling::Linear, DisplayDecimals::Two, None);
            Knob::new(cx, map.clone(), 0.5).on_changing(cx, |knob, cx|{
    
                // cx.emit(ParamChangeEvent::SetGain(knob.normalized_value));
                cx.emit(ParamChangeEvent::AllParams(1, knob.normalized_value))
            });
            Binding::new(cx, Params::params, move |cx, params|{
                Label::new(cx, &params.get(cx).res.get_display());
    
            });
        }).child_space(Stretch(1.0)).row_between(Pixels(10.0));

        VStack::new(cx, |cx|{
            Label::new(cx, "Drive");
            let map = GenericMap::new(0.0, 1.0, ValueScaling::Linear, DisplayDecimals::Two, None);
            Knob::new(cx, map.clone(), 0.5).on_changing(cx, |knob, cx|{
    
                // cx.emit(ParamChangeEvent::SetGain(knob.normalized_value));
                cx.emit(ParamChangeEvent::AllParams(2, knob.normalized_value))
            });
            Binding::new(cx, Params::params, move |cx, params|{
                Label::new(cx, &params.get(cx).drive.get_display());
    
            });
        }).child_space(Stretch(1.0)).row_between(Pixels(10.0));
        
        VStack::new(cx, |cx|{
            Binding::new(cx, Params::params, |cx, params|{
                let map = GenericMap::new(0.0, 1.0, ValueScaling::Linear, DisplayDecimals::Two, None);
                let ft = params.get(cx).filter_type.get();
                Label::new(cx, if ft == 0 {"Filter Mode"} else {"Slope"});

                Knob::new(cx, map.clone(), 0.5).on_changing(cx, move |knob, cx|{
        
                    // cx.emit(ParamChangeEvent::SetGain(knob.normalized_value));
                    cx.emit(ParamChangeEvent::AllParams(if ft == 0 {4} else {5}, knob.normalized_value))
                });
                Binding::new(cx, Params::params, move |cx, params|{
                    let ft = params.get(cx).filter_type.get();

                    Label::new(cx, &params.get(cx).get_parameter_text(if ft == 0 {4} else {5}));
        
                });

            })
        }).child_space(Stretch(1.0)).row_between(Pixels(10.0));
    }).background_color(Color::rgb(25, 25, 25)).child_space(Stretch(1.0)).row_between(Pixels(0.0));
    
    


}