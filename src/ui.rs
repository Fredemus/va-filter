use crate::editor::EditorState;
use crate::editor::{get_amplitude_response, get_phase_response};
use crate::filter;
use crate::filter_parameters::FilterParameterNr;
use crate::parameter::GetParameterByIndex;
use crate::utils::*;
use crate::FilterParameters;
use num_enum::FromPrimitive;
use femtovg::ImageFlags;
use femtovg::ImageId;
use femtovg::RenderTarget;
use femtovg::{Paint, Path};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use vizia::*;

use vst::host::Host;
use vst::plugin::HostCallback;
use vst::plugin::PluginParameters;
const ICON_DOWN_OPEN: &str = "\u{e75c}";
use std::f32::consts::PI;

#[derive(Lens)]
pub struct UiData {
    params: Arc<FilterParameters>,
    host: Option<HostCallback>,
    filter_circuits: Vec<String>,
    choice: String,
    show_phase: bool,
}

#[derive(Debug)]
pub enum ParamChangeEvent {
    AllParams(i32, f32),
    CircuitEvent(usize),
    ChangeBodeView(),
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

                ParamChangeEvent::CircuitEvent(index) => {
                    self.params.set_parameter(
                        FilterParameterNr::FilterType as i32,
                        *index as f32,
                    );

                    self.choice = self.filter_circuits[*index].clone()
                }
                ParamChangeEvent::ChangeBodeView() => {
                    self.show_phase = !self.show_phase;
                }
            }
        }
    }
}

pub fn plugin_gui(cx: &mut Context, state: Arc<EditorState>) {
    UiData {
        params: state.params.clone(),
        host: state.host,
        filter_circuits: vec![
            "State Variable Filter".to_string(),
            "Transistor Ladder".to_string(),
        ],
        choice: if state.params.filter_type.get() == 0 {
            "State Variable Filter".to_string()
        } else {
            "Transistor Ladder".to_string()
        },
        show_phase: false,
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
                    Label::new(cx, UiData::choice).left(Auto);
                    Label::new(cx, ICON_DOWN_OPEN).class("arrow");
                }),
                move |cx| {
                    // List of options
                    List::new(cx, UiData::filter_circuits, move |cx, index, item| {
                        VStack::new(cx, move |cx| {
                            Binding::new(cx, UiData::choice, move |cx, choice| {
                                let selected = *item.get(cx) == *choice.get(cx);
                                Label::new(cx, &item.get(cx).to_string())
                                    .width(Stretch(1.0))
                                    .class("item")
                                    .checked(selected)
                                    .on_press(move |cx| {
                                        cx.emit(ParamChangeEvent::CircuitEvent(index));
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
            make_knob(cx, FilterParameterNr::Cutoff as i32);
            // Resonance
            make_knob(cx, FilterParameterNr::Res as i32);
            // Drive
            make_knob(cx, FilterParameterNr::Drive as i32);
            // Mode/ Slope
            Binding::new(
                cx,
                UiData::params.map(|params| params.filter_type.get()),
                move |cx, ft| {
                    if *ft.get(cx) == 0 {
                        let param = &UiData::params.get(cx).mode;
                        let steps = (param.max - param.min + 1.) as usize;

                        make_steppy_knob(cx, FilterParameterNr::Mode as i32, steps, 270.);
                    } else {
                        let param = &UiData::params.get(cx).slope;
                        let steps = (param.max - param.min + 1.) as usize;
                        make_steppy_knob(cx, FilterParameterNr::Slope as i32, steps, 270.);
                    }
                },
            );
        })
        .class("knobs");

        BodePlot::new(cx)
            .class("bode")
            .text("Bode Plot")
            .overflow(Overflow::Visible)
            .on_press(|cx| {
                cx.emit(ParamChangeEvent::ChangeBodeView());
            });
    })
    .class("container");
}
// makes a knob linked to a parameter
fn make_knob(cx: &mut Context, param_index: i32) -> Handle<VStack> {
    VStack::new(cx, move |cx| {
        Label::new(
            cx,
            UiData::params.map(move |params| params.get_parameter_name(param_index)),
        );

        Knob::custom(
            cx,
            UiData::params
                .get(cx)
                .get_parameter_default(param_index),
                
            UiData::params.map(move |params| 
                
                params.get_parameter(param_index)),
            move |cx, lens| {
                TickKnob::new(
                    cx,
                    Percentage(80.0),
                    // Percentage(20.0),
                    Pixels(4.),
                    Percentage(50.0),
                    270.0,
                    KnobMode::Continuous,
                )
                .value(lens.clone())
                .class("tick");
                ArcTrack::new(
                    cx,
                    false,
                    Percentage(100.0),
                    Percentage(10.),
                    270.,
                    KnobMode::Continuous,
                )
                .value(lens)
                .class("track")
            },
        )
        .on_changing(move |cx, val| cx.emit(ParamChangeEvent::AllParams(param_index, val)));

        Label::new(
            cx,
            UiData::params.map(move |params| params.get_parameter_text(param_index)),
        );
    })
    .child_space(Stretch(1.0))
    .row_between(Pixels(10.0))
}
// using Knob::custom() to make a stepped knob with tickmarks indicating the steps
fn make_steppy_knob(
    cx: &mut Context,
    param_index: i32,
    steps: usize,
    arc_len: f32,
) -> Handle<VStack> {
    VStack::new(cx, move |cx| {
        Label::new(
            cx,
            UiData::params.map(move |params| params.get_parameter_name(param_index)),
        );

        Knob::custom(
            cx,
            UiData::params
                .get(cx)
                .get_parameter_by_index(param_index)
                .get_normalized_default(),
            UiData::params.map(move |params| 
                //params.get_parameter(param_index)),
                params.get_parameter(param_index)),
            move |cx, lens| {
                let mode = KnobMode::Discrete(steps);
                Ticks::new(
                    cx,
                    Percentage(100.0),
                    Percentage(25.0),
                    // Pixels(2.),
                    Pixels(2.0),
                    arc_len,
                    mode,
                )
                .class("track");
                TickKnob::new(
                    cx,
                    Percentage(80.0),
                    Pixels(4.),
                    Percentage(50.0),
                    arc_len,
                    mode,
                )
                .value(lens)
                .class("tick")
            },
        )
        .on_changing(move |cx, val| cx.emit(ParamChangeEvent::AllParams(param_index, val)));

        Label::new(
            cx,
            UiData::params.map(move |params| params.get_parameter_text(param_index)),
        );
    })
    .child_space(Stretch(1.0))
    .row_between(Pixels(10.0))
}

pub struct BodePlot {
    image: Rc<RefCell<Option<ImageId>>>,
}

impl BodePlot {
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self {
            image: Rc::new(RefCell::new(None)),
        }
        .build2(cx, |_| {})
    }
}

impl View for BodePlot {
    fn draw(&self, cx: &mut Context, canvas: &mut Canvas) {
        if let Some(ui_data) = cx.data::<UiData>() {
            let params = ui_data.params.clone();

            // TODO - Make this configurable
            let width = 360;
            let height = 200;

            let amps: Vec<f32>;
            let max;
            let min;
            //
            if ui_data.show_phase {
                if params.filter_type.get() == 0 {
                    let mode = params.mode.get();
                    amps = get_phase_response(
                        params.cutoff.get(),
                        params.zeta.get(),
                        mode,
                        params.filter_type.get(),
                        width,
                    );
                    if mode == 0 {
                        max = 0.;
                        // max phase shift of the state variable filter is Pi radians / 180 degrees
                        min = -PI;
                    } else if mode == 1 {
                        max = PI;
                        min = 0.;
                    } else {
                        max = PI / 2.;
                        min = -PI / 2.;
                    }
                } else {
                    amps = get_phase_response(
                        params.cutoff.get(),
                        // 2.,
                        params.k_ladder.get(),
                        params.slope.get(),
                        params.filter_type.get(),
                        width,
                    );
                    if params.slope.get() > 1 {
                        max = PI;
                        min = -PI;
                    } else {
                        max = PI / 2.;
                        min = -PI;
                    }
                };
            } else {
                // min and max amplitude values that will be rendered
                min = -60.0;
                max = 40.0;
                if params.filter_type.get() == 0 {
                    amps = get_amplitude_response(
                        params.cutoff.get(),
                        params.zeta.get(),
                        params.mode.get(),
                        params.filter_type.get(),
                        width,
                    );
                } else {
                    amps = get_amplitude_response(
                        params.cutoff.get(),
                        // 2.,
                        params.k_ladder.get(),
                        params.slope.get(),
                        params.filter_type.get(),
                        width,
                    );
                }
            }

            let bounds = cx.cache.get_bounds(cx.current);

            let image_id = if let Some(image_id) = *self.image.borrow() {
                image_id
            } else {
                canvas
                    .create_image_empty(
                        width,
                        height,
                        femtovg::PixelFormat::Rgb8,
                        ImageFlags::FLIP_Y,
                    )
                    .expect("Failed to create image")
            };

            *self.image.borrow_mut() = Some(image_id);

            canvas.set_render_target(RenderTarget::Image(image_id));

            let background_color = cx
                .style
                .background_color
                .get(cx.current)
                .cloned()
                .unwrap_or_default();
            let color = cx
                .style
                .font_color
                .get(cx.current)
                .cloned()
                .unwrap_or_default();

            // Fill background
            canvas.clear_rect(0, 0, width as u32, height as u32, background_color.into());

            let mut path = Path::new();
            let amp = amps[0].clamp(min, max);
            let y = height as f32 * ((amp - min) / (max - min));

            path.move_to(-10.0, height as f32 - y + 1.0);
            let line_width = 5.0;
            for i in 0..360 {
                let amp = amps[i].clamp(min, max);
                let y = height as f32 * ((amp - min) / (max - min));

                path.line_to(i as f32, height as f32 - y + line_width / 2.0);
            }

            let mut path2 = path.clone();
            // Draw plot
            let mut paint = Paint::color(color.into());
            paint.set_line_width(line_width);
            paint.set_line_join(femtovg::LineJoin::Round);
            paint.set_line_cap(femtovg::LineCap::Square);
            canvas.stroke_path(&mut path, paint);

            // making a cool background gradient
            let mut mid_color = femtovg::Color::from(color);
            mid_color.set_alpha(20);
            let mut edge_color = femtovg::Color::from(color);
            edge_color.set_alpha(64);
            // bg color is slightly less visible in the mid-point (0 dB) of the graph
            let bg = Paint::linear_gradient_stops(
                0.0,
                0.0,
                0.0,
                height as f32,
                // femtovg::Color::rgba(0, 160, 192, 0),
                // femtovg::Color::rgba(0, 160, 192, 64),
                &[(0.0, edge_color), (0.4, mid_color), (1.0, edge_color)],
            );
            // Making the background fill be contained by a line through the mid-point of the graph
            path2.line_to(width as f32, height as f32 * 0.4 + line_width / 2.0);
            path2.line_to(0., height as f32 * 0.4 + line_width / 2.0);
            canvas.fill_path(&mut path2, bg);

            canvas.set_render_target(RenderTarget::Screen);

            let mut path = Path::new();
            path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
            canvas.fill_path(
                &mut path,
                Paint::image(image_id, bounds.x, bounds.y, bounds.w, bounds.h, 0.0, 1.0),
            );
        }
    }
}
