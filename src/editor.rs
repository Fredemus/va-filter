// colorhexa.com is nice for looking for colors
// TODO: Instead of knobs beginning and ending changes on knob.value_changed, changes should be "grouped together"
// somehow, maybe by use of ui.is_item_active()?
// hmm, it seems our behavior (with regards to undoing at least) is the same as big boy problems, so not sure if it's an issue?
use imgui::*;
use imgui_knobs::*;

use crate::filter_parameters::FilterParameters;
use crate::utils::AtomicOps;
use imgui_baseview::{HiDpiMode, ImguiWindow, RenderSettings, Settings};

use crate::parameter::{ParameterF32, ParameterUsize};
use crate::vst::plugin::{HostCallback, PluginParameters};
use vst::host::Host;
mod plot;
use baseview::{Size, WindowOpenOptions, WindowScalePolicy};
use vst::editor::Editor;

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;

const WINDOW_WIDTH: usize = 512;
const WINDOW_HEIGHT: usize = 512;
const WINDOW_WIDTH_F: f32 = WINDOW_WIDTH as f32;
const WINDOW_HEIGHT_F: f32 = WINDOW_HEIGHT as f32;
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const CYAN: [f32; 4] = [0.1, 1.0, 1.0, 0.8];
const GREY: [f32; 4] = [0.1, 0.1, 0.1, 0.6];
// const SOLID_GREY: [f32; 4] = [GREY[0], GREY[1], GREY[2], 1.0];
// const BG_COLOR: [f32; 4] = [0.21 * 1.4, 0.11 * 1.7, 0.25 * 1.4, 1.0];
// const BG_COLOR_TRANSP: [f32; 4] = [0.21 * 1.4, 0.11 * 1.7, 0.25 * 1.4, 0.0];
// const GREEN: [f32; 4] = [0.23, 0.68, 0.23, 1.0];
// const RED: [f32; 4] = [0.98, 0.02, 0.22, 1.0];
const ORANGE: [f32; 4] = [1.0, 0.58, 0.0, 1.0];
const ORANGE_HOVERED: [f32; 4] = [1.0, 0.68, 0.1, 1.0];
const TEXT: [f32; 4] = [1.0, 1.0, 1.0, 0.75];

pub fn draw_knob(knob: &Knob, wiper_color: &ColorSet, track_color: &ColorSet) {
    knob.draw_arc(
        0.8,
        0.20,
        knob.angle_min,
        knob.angle_max,
        track_color,
        16,
        2,
    );
    if knob.t > 0.01 {
        knob.draw_arc(0.8, 0.21, knob.angle_min, knob.angle, wiper_color, 16, 2);
    }
}
/// keeps track of parameters and enables contact with the host
pub struct EditorState {
    pub params: Arc<FilterParameters>,
    pub host: Option<HostCallback>,
}
impl EditorState {
    fn draw_bode_plot(&self, ui: &Ui, size: [f32; 2]) {
        let draw_list = ui.get_window_draw_list();
        let cursor = ui.cursor_screen_pos();
        // adding a background
        draw_list
            .add_rect(
                [cursor[0], cursor[1] - size[1]],
                [cursor[0] + size[0], cursor[1]],
                GREY,
            )
            .filled(true)
            .thickness(5.)
            .build();

        let color = ORANGE;
        let mut amps: Vec<f32>;
        if self.params.filter_type.get() == 0 {
            amps = plot::get_filter_bode(
                self.params.cutoff.get(),
                self.params.zeta.get(),
                self.params.mode.get(),
                self.params.filter_type.get(),
            );
        } else {
            amps = plot::get_filter_bode(
                self.params.cutoff.get(),
                self.params.k_ladder.get(),
                self.params.slope.get(),
                self.params.filter_type.get(),
            );
        };

        let maxmin = 40.;
        // normalizing amplitudes
        for x in &mut amps {
            *x = (*x - (-maxmin)) / (maxmin - (-maxmin))
        }
        let length = amps.len();
        let scale = (size[0] / length as f32) as f32;
        // let scale_y = size[1] / 2.;
        let scale_y = size[1];
        let mut last = amps[0] * scale_y;
        for i in 1..length {
            // The scale might give problems with clipping out if resonance is higher than +12 dB
            let next = amps[i] * scale_y;

            let fi = i as f32;
            // only draw values that are within bounds
            if last > 0. && next < scale_y {
                //draw line from i to i+1
                draw_list
                    .add_line(
                        [cursor[0] + fi * scale, cursor[1] - last],
                        [cursor[0] + fi * scale + 1., cursor[1] - next],
                        color,
                    )
                    .thickness(5.)
                    .build();
            }

            last = next;
        }
        // adding a frame that covers up some weird stuff with end lines
        draw_list
            .add_rect(
                [cursor[0], cursor[1] - size[1]],
                [cursor[0] + size[0], cursor[1]],
                BLACK,
            )
            .filled(false)
            .thickness(8.)
            .build();
    }
    // Todo: Potentially we can avoid passing in parameter reference and just use parameter_index to get stuff we need
    pub fn make_knob(
        &self,
        ui: &Ui,
        parameter: &ParameterF32,
        parameter_index: i32,
        wiper_color: &ColorSet,
        track_color: &ColorSet,
        title_fix: f32,
    ) {
        let width = ui.text_line_height() * 4.75;
        let w = ui.push_item_width(width);
        // let title = parameter.get_name();
        let title = parameter.get_name();
        let knob_id = &ImString::new(format!("##{}_KNOB_CONTROL_", title));
        knob_title(ui, &ImString::new(title.to_uppercase()), width);
        let cursor = ui.cursor_pos();
        ui.set_cursor_pos([cursor[0], cursor[1] + 5.0]);
        let mut val = parameter.get_normalized();
        let knob = Knob::new(
            ui,
            knob_id,
            &mut val,
            // 0.,
            // 1.,
            (parameter.get_func)(parameter.from_range(parameter.default)),
            width * 0.5,
            true,
        );
        let cursor = ui.cursor_pos();
        ui.set_cursor_pos([cursor[0] + title_fix, cursor[1] - 10.0]);
        knob_title(ui, &ImString::new(parameter.get_display()), width);
        // knob_title(ui, &ImString::new(format!("v:{:.1} a:{:.1}", parameter.get_normalized(), knob.angle)), width); // for testing

        if knob.value_changed {
            self.params.set_parameter(parameter_index, *knob.p_value);
            // if the filter is hosted, inform the host that gui has changed parameters
            // Option around host could be removed if we don't want a standalone version
            if let Some(host) = self.host {
                // Todo: Is there a better way to end and begin edit?
                // Also, it seems that automate sets the parameter (in Ableton at least)
                // so 2 set_parameter calls seems wasteful, but without both, the parameters end up not-quite-right
                host.begin_edit(parameter_index);
                host.automate(parameter_index, *knob.p_value);
                host.end_edit(parameter_index);
            }
        }

        w.pop(ui);
        draw_knob(&knob, wiper_color, track_color);
    }
    /// Meant for knobs that go through discrete steps.
    pub fn make_steppy_knob(
        &self,
        ui: &Ui,
        parameter: &ParameterUsize,
        parameter_index: i32,
        wiper_color: &ColorSet,
        track_color: &ColorSet,
        title_fix: f32,
    ) {
        let width = ui.text_line_height() * 4.75;
        let w = ui.push_item_width(width);
        // let title = parameter.get_name();
        let title = parameter.get_name();
        let knob_id = &ImString::new(format!("##{}_KNOB_CONTORL_", title));
        knob_title(ui, &ImString::new(title.clone().to_uppercase()), width);
        let cursor = ui.cursor_pos();
        ui.set_cursor_pos([cursor[0], cursor[1] + 5.0]);
        let mut val = parameter.get_normalized();
        let knob = Knob::new(
            ui,
            knob_id,
            &mut val,
            // 0.,
            // 1.,
            (parameter.get_func)(parameter.from_range(parameter.default as f32)),
            width * 0.5,
            true,
        );
        let cursor = ui.cursor_pos();
        ui.set_cursor_pos([cursor[0] + title_fix, cursor[1] - 10.0]);
        knob_title(ui, &ImString::new(parameter.get_display()), width);

        if knob.value_changed {
            self.params.set_parameter(parameter_index, *knob.p_value);
            // if the filter is hosted, inform the host that gui has changed parameters
            // Option around host could be removed if we don't want a standalone version
            if let Some(host) = self.host {
                // Todo: Is there a better way to end and begin edit?
                // Also, it seems that automate sets the parameter (in Ableton at least)
                // so 2 set_parameter calls seems wasteful, but without both, the parameters end up not-quite-right
                host.begin_edit(parameter_index);
                host.automate(parameter_index, *knob.p_value);
                host.end_edit(parameter_index);
            }
        }
        w.pop(ui);
        draw_stepped_knob(
            &knob,
            (parameter.max - parameter.min + 1.) as u32,
            wiper_color,
            track_color,
            &ColorSet::from(CYAN),
        );
    }
}
pub struct SVFPluginEditor {
    pub is_open: bool,
    pub state: Arc<EditorState>,
}
fn move_cursor(ui: &Ui, x: f32, y: f32) {
    let cursor = ui.cursor_pos();
    ui.set_cursor_pos([cursor[0] + x, cursor[1] + y])
}

fn _floating_text(ui: &Ui, text: &str) {
    ui.get_window_draw_list()
        .add_text(ui.cursor_pos(), ui.style_color(StyleColor::Text), text)
}
impl Editor for SVFPluginEditor {
    fn position(&self) -> (i32, i32) {
        (0, 0)
    }

    fn size(&self) -> (i32, i32) {
        (WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32)
    }

    fn open(&mut self, parent: *mut ::std::ffi::c_void) -> bool {
        //::log::info!("self.running {}", self.running);
        if self.is_open {
            return false;
        }

        self.is_open = true;

        let settings = Settings {
            window: WindowOpenOptions {
                title: String::from("filter boy window"),
                size: Size::new(WINDOW_WIDTH as f64, WINDOW_HEIGHT as f64),
                scale: WindowScalePolicy::SystemScaleFactor,
            },
            clear_color: (0.0, 0.0, 0.0),
            hidpi_mode: HiDpiMode::Default,
            render_settings: RenderSettings::default(),
        };

        ImguiWindow::open_parented(
            &VstParent(parent),
            settings,
            self.state.clone(),
            |ctx: &mut Context, _state: &mut Arc<EditorState>| {
                ctx.fonts().add_font(&[FontSource::TtfData {
                    data: include_bytes!("../OpenSans-Semibold.ttf"),
                    size_pixels: 20.0,
                    config: None,
                }]);
            },
            |_run: &mut bool, ui: &Ui, state: &mut Arc<EditorState>| {
                // {
                //     let mut editor_only = state.editor_only.lock().unwrap();
                //     editor_only.sample_data.consume();
                // }
                //ui.show_demo_window(run);
                let w = Window::new(im_str!("does this matter?"))
                    .size([WINDOW_WIDTH_F, WINDOW_HEIGHT_F], Condition::Appearing)
                    .position([0.0, 0.0], Condition::Appearing)
                    .draw_background(false)
                    .no_decoration()
                    .movable(false);
                w.build(&ui, || {
                    let text_style_color = ui.push_style_color(StyleColor::Text, TEXT);

                    ui.set_cursor_pos([0.0, 25.0]);

                    let highlight = ColorSet::new(ORANGE, ORANGE_HOVERED, ORANGE_HOVERED);

                    let params = &state.params;

                    let _line_height = ui.text_line_height();
                    let n_columns = 5;
                    let lowlight = ColorSet::from(BLACK);
                    ui.columns(n_columns, im_str!("cols"), false);
                    let width = WINDOW_WIDTH_F / n_columns as f32 - 0.25;
                    for i in 1..n_columns {
                        ui.set_column_width(i, width);
                    }
                    ui.set_column_width(0, width * 0.5);

                    ui.next_column();
                    state.make_knob(ui, &params.cutoff, 0, &highlight, &lowlight, 0.0);
                    move_cursor(ui, 0.0, -113.0);

                    ui.next_column();

                    state.make_knob(ui, &params.res, 1, &highlight, &lowlight, 0.0);
                    ui.next_column();

                    state.make_knob(ui, &params.drive, 2, &highlight, &lowlight, 0.0);
                    ui.next_column();
                    if params.filter_type.get() == 0 {
                        state.make_steppy_knob(ui, &params.mode, 4, &highlight, &lowlight, 0.0);
                    } else {
                        state.make_steppy_knob(ui, &params.slope, 5, &highlight, &lowlight, 0.0);
                    }
                    ui.next_column();

                    ui.columns(1, im_str!("nocols"), false);
                    // move_cursor(ui, (WINDOW_WIDTH_F - 400.) / 2., 333.);
                    // TODO: I would love if this cursor pos could come from knob size or smth
                    ui.set_cursor_pos([(WINDOW_WIDTH_F - 400.) / 2., WINDOW_HEIGHT_F - 4.]);

                    state.draw_bode_plot(ui, [400., 335.]);

                    text_style_color.pop(ui);
                });
            },
        );

        true
    }

    fn is_open(&mut self) -> bool {
        self.is_open
    }

    fn close(&mut self) {
        self.is_open = false;
    }
}
struct VstParent(*mut ::std::ffi::c_void);

#[cfg(target_os = "macos")]
unsafe impl HasRawWindowHandle for VstParent {
    fn raw_window_handle(&self) -> RawWindowHandle {
        use raw_window_handle::macos::MacOSHandle;

        RawWindowHandle::MacOS(MacOSHandle {
            ns_view: self.0 as *mut ::std::ffi::c_void,
            ..MacOSHandle::empty()
        })
    }
}

#[cfg(target_os = "windows")]
unsafe impl HasRawWindowHandle for VstParent {
    fn raw_window_handle(&self) -> RawWindowHandle {
        use raw_window_handle::windows::WindowsHandle;

        RawWindowHandle::Windows(WindowsHandle {
            hwnd: self.0,
            ..WindowsHandle::empty()
        })
    }
}

#[cfg(target_os = "linux")]
unsafe impl HasRawWindowHandle for VstParent {
    fn raw_window_handle(&self) -> RawWindowHandle {
        use raw_window_handle::unix::XcbHandle;

        RawWindowHandle::Xcb(XcbHandle {
            window: self.0 as u32,
            ..XcbHandle::empty()
        })
    }
}

#[test]
#[ignore]
fn spawn_gui() {
    let params = Arc::new(FilterParameters::default());
    let editor = SVFPluginEditor {
        is_open: false,
        state: Arc::new(EditorState {
            params: params,
            host: None,
        }),
    };
    let settings = Settings {
        window: WindowOpenOptions {
            title: String::from("synthboy window"),
            size: Size::new(WINDOW_WIDTH as f64, WINDOW_HEIGHT as f64),
            scale: WindowScalePolicy::SystemScaleFactor,
        },
        clear_color: (0.0, 0.0, 0.0),
        hidpi_mode: HiDpiMode::Default,
        render_settings: RenderSettings::default(),
    };

    ImguiWindow::open_blocking(
        settings,
        editor.state.clone(),
        |ctx: &mut Context, _state: &mut Arc<EditorState>| {
            ctx.fonts().add_font(&[FontSource::TtfData {
                data: include_bytes!("../OpenSans-Semibold.ttf"),
                size_pixels: 20.0,
                config: None,
            }]);
        },
        |_run: &mut bool, ui: &Ui, state: &mut Arc<EditorState>| {
            let w = Window::new(im_str!("does this matter?"))
                    .size([WINDOW_WIDTH_F, WINDOW_HEIGHT_F], Condition::Appearing)
                    .position([0.0, 0.0], Condition::Appearing)
                    .draw_background(false)
                    .no_decoration()
                    .movable(false);
                w.build(&ui, || {
                    let text_style_color = ui.push_style_color(StyleColor::Text, TEXT);

                    ui.set_cursor_pos([0.0, 25.0]);

                    let highlight = ColorSet::new(ORANGE, ORANGE_HOVERED, ORANGE_HOVERED);

                    let params = &state.params;

                    let _line_height = ui.text_line_height();
                    let n_columns = 5;
                    let lowlight = ColorSet::from(BLACK);
                    ui.columns(n_columns, im_str!("cols"), false);
                    let width = WINDOW_WIDTH_F / n_columns as f32 - 0.25;
                    for i in 1..n_columns {
                        ui.set_column_width(i, width);
                    }
                    ui.set_column_width(0, width * 0.5);

                    ui.next_column();
                    state.make_knob(ui, &params.cutoff, 0, &highlight, &lowlight, 0.0);
                    move_cursor(ui, 0.0, -113.0);

                    ui.next_column();

                    state.make_knob(ui, &params.res, 1, &highlight, &lowlight, 0.0);
                    ui.next_column();

                    state.make_knob(ui, &params.drive, 2, &highlight, &lowlight, 0.0);
                    ui.next_column();
                    if params.filter_type.get() == 0 {
                        state.make_steppy_knob(ui, &params.mode, 4, &highlight, &lowlight, 0.0);
                    } else {
                        state.make_steppy_knob(ui, &params.slope, 5, &highlight, &lowlight, 0.0);
                    }
                    ui.next_column();

                    ui.columns(1, im_str!("nocols"), false);
                    // move_cursor(ui, (WINDOW_WIDTH_F - 400.) / 2., 333.);
                    // TODO: I would love if this cursor pos could come from knob size or smth
                    ui.set_cursor_pos([(WINDOW_WIDTH_F - 400.) / 2., WINDOW_HEIGHT_F - 4.]);

                    state.draw_bode_plot(ui, [400., 335.]);

                    text_style_color.pop(ui);
            });
        },
    );
}

