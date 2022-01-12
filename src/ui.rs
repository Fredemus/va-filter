use crate::editor::EditorState;
use crate::utils::*;
use crate::FilterParameters;
use std::sync::Arc;
use vizia::*;
use vst::host::Host;
use vst::plugin::HostCallback;
use vst::plugin::PluginParameters;
const ICON_DOWN_OPEN: &str = "\u{e75c}";

#[derive(Lens)]
pub struct UiData {
    params: Arc<FilterParameters>,
    host: Option<HostCallback>,
    filter_circuits: Vec<String>,
    choice: String,
}

#[derive(Debug)]
pub enum ParamChangeEvent {
    AllParams(i32, f32),
    CircuitEvent(String),
}

impl Model for UiData {
    fn event(&mut self, _cx: &mut Context, event: &mut Event) {
        if let Some(param_change_event) = event.message.downcast() {
            match param_change_event {
                ParamChangeEvent::AllParams(parameter_index, new_value) => {
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
                    } else {
                        self.params.set_parameter(3, 1.);
                    }
                    self.choice = circuit_name.to_string();
                }
            }
        }
    }
}

pub fn plugin_gui(cx: &mut Context, state: Arc<EditorState>) {

    UiData {
        params: state.params.clone(),
        host: state.host,
        filter_circuits: vec!["SVF".to_string(), "Transistor Ladder".to_string()],
        choice: if state.params.filter_type.get() == 0 {
            "SVF".to_string()
        } else {
            "Transistor Ladder".to_string()
        },
    }
    .build(cx);

    VStack::new(cx, |cx| {
        // Filter circuit selection
        HStack::new(cx, |cx| {
            Label::new(cx, "Filter Circuit");
            // Dropdown to select filter circuit
            Dropdown::new(
                cx,
                move |cx|
                // A Label and an Icon
                HStack::new(cx, move |cx|{
                    //let choice = choice.get(cx).clone();
                    Binding::new(cx, UiData::choice, |cx, choice|{
                        Label::new(cx, &choice.get(cx).to_string()).left(Auto);
                    });
                    Label::new(cx, ICON_DOWN_OPEN).class("arrow");
                }),
                move |cx| {
                    // List of options
                    List::new(cx, UiData::filter_circuits, move |cx, item| {
                        VStack::new(cx, move |cx| {
                            Binding::new(cx, UiData::choice, move |cx, choice|{
                                let selected = *item.get(cx) == *choice.get(cx);
                                Label::new(cx, &item.get(cx).to_string())
                                    .width(Stretch(1.0))
                                    .background_color(if selected {Color::from("#f8ac14")} else {Color::transparent()})
                                    .on_press(move |cx| {
                                        cx.emit(ParamChangeEvent::CircuitEvent(item.get(cx).clone()));
                                        cx.emit(PopupEvent::Close);
                                    });
                            });
                        });
                    });
                },
            );
            
        })
        .class("circuit_selector");

        // The filter control knobs
        HStack::new(cx, |cx| {
            // Cutoff
            make_knob(cx, 0);
            // Resonance
            make_knob(cx, 1);
            // Drive
            make_knob(cx, 2);
            // Mode/ Slope
            Binding::new(cx, UiData::params, move |cx, params| {
                let ft = params.get(cx).filter_type.get();
                if ft == 0 {
                    make_knob(cx, 4);
                } else {
                    make_knob(cx, 5);
                }
            });
        })
        .class("knobs");

        // Placeholder for bode plot
        Element::new(cx).class("bode").text("Bode Plot");
    })
    .class("container");
}

fn make_knob(cx: &mut Context, param_index: i32) -> Handle<VStack> {
    VStack::new(cx, move |cx| {
        Binding::new(cx, UiData::params, move |cx, params| {
            Label::new(cx, &params.get(cx).get_parameter_name(param_index));
            Knob::new(
                cx,
                params.get(cx)._get_parameter_default(param_index),
                params.get(cx).get_parameter(param_index),
                false,
            )
            .on_changing(move |knob, cx| {
                cx.emit(ParamChangeEvent::AllParams(
                    param_index,
                    knob.normalized_value,
                ))
            });
            Label::new(cx, &params.get(cx).get_parameter_text(param_index));
        });
    })
    .child_space(Stretch(1.0))
    .row_between(Pixels(10.0))
}