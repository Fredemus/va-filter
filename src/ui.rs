// use crate::editor::EditorState;
mod plot;
use crate::filter_params_nih::Circuits;
use nih_plug::context::GuiContext;
use nih_plug::param::internals::ParamPtr;
use plot::{get_amplitude_response, get_phase_response};
// use crate::editor::{get_amplitude_response, get_phase_response};
use crate::utils::*;
use crate::FilterParams;
use femtovg::ImageFlags;
use femtovg::ImageId;
use femtovg::RenderTarget;
use femtovg::{Paint, Path};
use nih_plug::prelude::Param;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use vizia::*;
// use vst::host::Host;
// use vst::plugin::HostCallback;
// use vst::plugin::PluginParameters;
const ICON_DOWN_OPEN: &str = "\u{e75c}";

use std::f32::consts::PI;

#[derive(Lens)]
pub struct UiData {
    pub gui_context: Arc<dyn GuiContext>,
    params: Arc<FilterParams>,
    // host: Option<HostCallback>,
    filter_circuits: Vec<String>,
    choice: String,
    show_phase: bool,
}

#[derive(Debug)]
pub enum ParamChangeEvent {
    BeginSet(ParamPtr),
    EndSet(ParamPtr),
    SetParam(ParamPtr, f32),

    CircuitEvent(String),
    ChangeBodeView(),
}

impl Model for UiData {
    fn event(&mut self, _cx: &mut Context, event: &mut Event) {
        // let setter = ParamSetter::new(self.gui_context.as_ref());
        event.map(|event, _| match event {
            ParamChangeEvent::SetParam(param_ptr, new_value) => {
                unsafe {
                    self.gui_context
                        .raw_set_parameter_normalized(*param_ptr, *new_value)
                };
            }

            ParamChangeEvent::BeginSet(param_ptr) => {
                unsafe { self.gui_context.raw_begin_set_parameter(*param_ptr) };
            }
            ParamChangeEvent::EndSet(param_ptr) => {
                unsafe { self.gui_context.raw_end_set_parameter(*param_ptr) };
            }
            ParamChangeEvent::CircuitEvent(circuit_name) => {
                self.choice = circuit_name.to_owned();
                if circuit_name == "SVF" {
                    unsafe {
                        self.gui_context
                            .raw_set_parameter_normalized(self.params.filter_type.as_ptr(), 0.)
                    };
                } else {
                    // self.params.set_parameter(3, 1.);
                    unsafe {
                        self.gui_context
                            .raw_set_parameter_normalized(self.params.filter_type.as_ptr(), 1.)
                    };
                }
            }
            ParamChangeEvent::ChangeBodeView() => {
                self.show_phase = !self.show_phase;
            }
        })
    }
}

pub fn plugin_gui(cx: &mut Context, params: Arc<FilterParams>, context: Arc<dyn GuiContext>) {
    UiData {
        gui_context: context.clone(),
        params: params.clone(),
        // host: state.host,
        filter_circuits: vec!["SVF".to_string(), "Transistor Ladder".to_string()],
        choice: if params.filter_type.value() == Circuits::SVF {
            "SVF".to_string()
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
                }).class("title"),
                move |cx| {
                    // List of options
                    List::new(cx, UiData::filter_circuits, move |cx, _, item| {
                        VStack::new(cx, move |cx| {
                            Binding::new(cx, UiData::choice, move |cx, choice| {
                                let selected = *item.get(cx) == *choice.get(cx);
                                Label::new(cx, &item.get(cx).to_string())
                                    .width(Stretch(1.0))
                                    .background_color(if selected {
                                        Color::from("#c28919")
                                    } else {
                                        Color::transparent()
                                    })
                                    .on_press(move |cx| {
                                        cx.emit(ParamChangeEvent::CircuitEvent(
                                            item.get(cx).clone(),
                                        ));
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
            // UiData::params.filter_type;
            // Cutoff
            // make_knob(cx, 0);
            make_knob(cx, params.cutoff.as_ptr(), |params| &params.cutoff);
            // Resonance
            // make_knob(cx, 1);
            make_knob(cx, params.res.as_ptr(), |params| &params.res);
            // // Drive
            make_knob(cx, params.drive.as_ptr(), |params| &params.drive);
            // Mode/ Slope
            Binding::new(
                cx,
                UiData::params.map(|params| params.filter_type.value() as usize),
                move |cx, ft| {
                    if ft.get(cx) == 0 {
                        // let param = &UiData::params.get(cx).mode;
                        // let steps = (param.max - param.min + 1.) as usize;
                        let steps = 5;
                        make_steppy_knob(cx, steps, 270., params.mode.as_ptr(), |params| {
                            &params.mode
                        });
                    } else {
                        let steps = 4;
                        make_steppy_knob(cx, steps, 270., params.slope.as_ptr(), |params| {
                            &params.slope
                        });
                        // let param = &UiData::params.get(cx).slope;
                        // let steps = (param.max - param.min + 1.) as usize;
                        // make_steppy_knob(cx, 5, steps, 270.);
                    }
                },
            );
        })
        .class("knobs");

        BodePlot::new(cx).class("bode").on_press(|cx| {
            cx.emit(ParamChangeEvent::ChangeBodeView());
        });
    })
    .class("container");
}
// makes a knob linked to a parameter
// fn make_knob<'a, P: Param>(cx: &mut Context, param: &'a P, setter: &'a ParamSetter<'a>) // -> Handle<VStack>
fn make_knob<P, F>(cx: &mut Context, param_ptr: ParamPtr, params_to_param: F) -> Handle<VStack>
where
    P: Param,
    F: 'static + Fn(&Arc<FilterParams>) -> &P + Copy,
    // L: Lens<Target = ParamPtr>,
{
    VStack::new(cx, move |cx| {
        // doesn't need to be a lens
        Label::new(
            cx,
            UiData::params.map(move |params| params_to_param(params).name().to_owned()),
        );

        Knob::custom(
            cx,
            // UiData::params.get(cx).get_parameter_default(param_index),
            0.5,
            // params.get(cx).get_parameter(param_index),
            UiData::params.map(move |params| params_to_param(params).normalized_value()),
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
                    -135.,
                    135.,
                    KnobMode::Continuous,
                )
                .value(lens)
                .class("track")
            },
        )
        .on_changing(move |cx, val| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                // ParamChangeEvent::AllParams(param_index, val),
                ParamChangeEvent::SetParam(param_ptr, val),
            )
        })
        .on_press(move |cx| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                ParamChangeEvent::BeginSet(param_ptr),
            )
        })
        .on_release(move |cx| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                ParamChangeEvent::EndSet(param_ptr),
            )
        });

        Label::new(
            cx,
            UiData::params.map(move |params| params_to_param(params).to_string()),
        )
        .width(Pixels(100.));
    })
    .child_space(Stretch(1.0))
    .row_between(Pixels(10.0))
}
// using Knob::custom() to make a stepped knob with tickmarks indicating the steps
fn make_steppy_knob<'a, P, F>(
    cx: &mut Context,
    steps: usize,
    arc_len: f32,
    param_ptr: nih_plug::param::internals::ParamPtr,
    params_to_param: F,
) where
    P: Param,
    F: 'static + Fn(&Arc<FilterParams>) -> &P + Copy,
{
    VStack::new(cx, move |cx| {
        Label::new(
            cx,
            UiData::params.map(move |params| params_to_param(params).name().to_owned()),
        );

        Knob::custom(
            cx,
            0.5,
            // UiData::params.map(move |params| {
            //     params.get_parameter(param_index)
            // }),
            UiData::params.map(move |params| params_to_param(params).normalized_value()),
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
        .on_changing(move |cx, val| cx.emit(ParamChangeEvent::SetParam(param_ptr, val)))
        .on_press(move |cx| cx.emit(ParamChangeEvent::BeginSet(param_ptr)))
        .on_release(move |cx| cx.emit(ParamChangeEvent::EndSet(param_ptr)));

        Label::new(
            cx,
            // UiData::params.map(move |params| params.get_parameter_text(param_index)),
            UiData::params.map(move |params| params_to_param(params).to_string()),
        )
        .width(Pixels(100.));
    })
    .child_space(Stretch(1.0))
    .row_between(Pixels(10.0));
}

pub struct BodePlot {
    image: Rc<RefCell<Option<ImageId>>>,
}

impl BodePlot {
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self {
            image: Rc::new(RefCell::new(None)),
        }
        .build(cx, |_| {})
    }
}

impl View for BodePlot {
    fn draw(&self, cx: &mut DrawContext<'_>, canvas: &mut Canvas) {
        let current = cx.current();
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
                if params.filter_type.value() == Circuits::SVF {
                    let mode = params.mode.value() as usize;
                    amps = get_phase_response(
                        params.cutoff.value,
                        params.zeta.get(),
                        mode,
                        params.filter_type.value() as usize,
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
                        params.cutoff.value,
                        // 2.,
                        params.k_ladder.get(),
                        params.slope.value() as usize,
                        params.filter_type.value() as usize,
                        width,
                    );
                    if params.slope.value() as usize > 1 {
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
                if params.filter_type.value() == Circuits::SVF {
                    amps = get_amplitude_response(
                        params.cutoff.value,
                        params.zeta.get(),
                        params.mode.value() as usize,
                        params.filter_type.value() as usize,
                        width,
                    );
                } else {
                    amps = get_amplitude_response(
                        params.cutoff.value,
                        // 2.,
                        params.k_ladder.get(),
                        params.slope.value() as usize,
                        params.filter_type.value() as usize,
                        width,
                    );
                }
            }

            let bounds = cx.cache().get_bounds(current);

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

            let background_color: femtovg::Color = cx
                .background_color(current)
                .cloned()
                .unwrap_or_default()
                .into();
            let color: femtovg::Color = cx.font_color(current).cloned().unwrap_or_default().into();

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
            canvas.save();
            canvas.reset_scissor();
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

            canvas.restore();
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
