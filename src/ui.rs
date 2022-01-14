use crate::editor::EditorState;
use crate::editor::get_filter_bode;
use crate::utils::*;
use crate::FilterParameters;
use std::sync::Arc;
use femtovg::ImageFlags;
use femtovg::ImageId;
use femtovg::RenderTarget;
use vizia::*;
use femtovg::{Path, Paint};

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
        //Element::new(cx).class("bode").text("Bode Plot");

        BodePlot::new(cx).class("bode").text("Bode Plot").overflow(Overflow::Visible);

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

pub struct BodePlot {
    image: Option<ImageId>,
}

impl BodePlot {
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self {
            image: None,
        }.build2(cx, |_|{

        })
    } 
}

impl View for BodePlot {
    fn draw(&self, cx: &Context, canvas: &mut Canvas) {

        if let Some(ui_data) = cx.data::<UiData>() {
            let params = ui_data.params.clone();

            let mode_or_slope = if params.filter_type.get() == 0 {
                params.mode.get()
            } else {
                params.slope.get()
            };

            let amps = get_filter_bode(
                params.cutoff.get(),
                1.0,
                mode_or_slope,
                params.filter_type.get());
    
            let bounds = cx.cache.get_bounds(cx.current);

            // TODO - Make this configurable
            let width = 360;
            let height = 200;

            let image_id = if let Some(image_id) = self.image {
                image_id
            } else {
                canvas.create_image_empty(width, height, femtovg::PixelFormat::Rgb8, ImageFlags::FLIP_Y).expect("Failed to create image")
            };

            

            canvas.set_render_target(RenderTarget::Image(image_id));

            let background_color = cx.style.background_color.get(cx.current).cloned().unwrap_or_default();
            let color = cx.style.font_color.get(cx.current).cloned().unwrap_or_default();
            

            // Fill background
            canvas.clear_rect(0, 0, width as u32, height as u32, background_color.into());
            
            let min = -60.0;
            let max = 40.0;

            let mut path = Path::new();
            let amp = amps[0].clamp(min, max);
            let y = height as f32 * ((amp - min) / (max - min)); 

            path.move_to(0.0, height as f32 - y + 1.0);

            for i in 1..360 {
                let amp = amps[i].clamp(min, max);
                let y = height as f32 * ((amp - min) / (max - min));
    
                path.line_to(i as f32, height as f32 - y + 1.0);
            }

            // Draw plot
            let mut paint = Paint::color(color.into());
            paint.set_line_width(2.0);
            canvas.stroke_path(&mut path, paint);


            canvas.set_render_target(RenderTarget::Screen);

            let mut path = Path::new();
            path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
            canvas.fill_path(&mut path, Paint::image(image_id, bounds.x, bounds.y, bounds.w, bounds.h, 0.0, 1.0));
        
        }



    }
}