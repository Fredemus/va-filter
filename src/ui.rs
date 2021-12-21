use std::sync::Arc;
use crate::utils::*;
use vizia::*;
use vst::host::Host;
use vst::plugin::HostCallback;
use vst::plugin::PluginParameters;
use crate::editor::EditorState;
use crate::FilterParameters;
const ICON_DOWN_OPEN: &str = "\u{e75c}";
const STYLE: &str = r#"
    dropdown .title {
        background-color: #101010;
        height: 30px;
        width: 100px;
        child-space: 1s;
        child-left: 5px;
    }
    dropdown>popup {
        background-color: #141414;
    }
    button {
        width: auto;
        height: auto;
        child-space: 5px;
        background-color: gray;
    }
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
pub struct UiData {
    params: Arc<FilterParameters>,
    host: Option<HostCallback>,
    filter_circuits: Vec<String>,
    choice: String,
}

#[derive(Debug)]
pub enum ParamChangeEvent {
    _SetGain(f32),
    AllParams(i32, f32),
    CircuitEvent(String),
}

impl Model for UiData {
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
                ParamChangeEvent::CircuitEvent(circuit_name) => {
                    if circuit_name == "SVF" {
                        self.params.set_parameter(3, 0.);
                    }
                    else {
                        self.params.set_parameter(3, 1.);
                    }
                    self.choice = circuit_name.to_string();
                }
            }
        }
    }
}


pub fn plugin_gui(cx: &mut Context, state: Arc<EditorState> ) {
    cx.add_theme(STYLE);

    UiData {
        params: state.params.clone(),
        host: state.host,
        filter_circuits: vec!["SVF".to_string(), "Transistor ladder".to_string()],
        choice: "SVF".to_string(),
    }.build(cx);
    VStack::new(cx, |cx|{
        // dropdown to select filter circuit
        Binding::new(cx, UiData::choice, |cx, choice|{
            // Dropdown List
            Dropdown::new(cx, move |cx|
                // A Label and an Icon
                HStack::new(cx, move |cx|{
                    let choice = "Choose filter circuit".to_string();
                    Label::new(cx, &choice);
                    Label::new(cx, ICON_DOWN_OPEN).font("icons").left(Stretch(1.0)).right(Pixels(5.0));
                }), 
                move |cx|{
                // List of options
                List::new(cx, UiData::filter_circuits, move |cx, item|{
                    // Need this because of a bug to do ith bindings inside a list
                    VStack::new(cx, move |cx|{
                            let option = item.get(cx).clone();
                            // Button which updates the chosen option
                            Button::new(cx, move |cx| {
                                cx.emit(ParamChangeEvent::CircuitEvent(option.clone()));
                                cx.emit(PopupEvent::Close);
                            }, move |cx|{
                                let opt = item.get(cx).clone();
                                Label::new(cx, &opt.clone()).width(Stretch(1.0)).height(Pixels(20.0))
                            }).width(Stretch(1.0)).background_color(if item.get(cx) == choice.get(cx) {Color::from("#f8ac14")} else {Color::transparent()});
                    }).width(Stretch(1.0));
                });
            }).z_order(100);
            });
        // The knobs
        HStack::new(cx, |cx| {
            // TODO: using a width here seems to make everything not-centered
            let knob_width = Pixels(90.);
            make_knob(cx, 0, knob_width);
            // resonance
            make_knob(cx, 1, knob_width);
            //drive
            make_knob(cx, 2, knob_width);
            Binding::new(cx, UiData::params, move |cx, params|{
                let ft = params.get(cx).filter_type.get();
                if ft == 0 {
                    make_knob(cx, 4, knob_width);
                }
                else {
                    make_knob(cx, 5, knob_width);
                }
            });
            
        }).child_space(Pixels(120.)).row_between(Pixels(0.0));
    }).background_color(Color::rgb(25, 25, 25)).child_space(Stretch(1.0)).row_between(Pixels(10.0));
    

}

fn make_knob(cx: &mut Context, param_index: i32, width: Units) {
    VStack::new(cx, move |cx|{
        Binding::new(cx, UiData::params, move |cx, params|{
            Label::new(cx, &params.get(cx).get_parameter_name(param_index));
            Knob::new(cx, params.get(cx)._get_parameter_default(param_index), params.get(cx).get_parameter(param_index), false).on_changing(cx, move |knob, cx,|{
                cx.emit(ParamChangeEvent::AllParams(param_index, knob.normalized_value))
            });
            Label::new(cx, &params.get(cx).get_parameter_text(param_index)).height(Pixels(20.));
        });
    }).child_space(Stretch(1.0)).row_between(Pixels(10.0)).width(width);
}