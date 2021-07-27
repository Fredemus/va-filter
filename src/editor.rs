use imgui::*;
use imgui_knobs::*;

use imgui_baseview::{HiDpiMode, ImguiWindow, RenderSettings, Settings};
// use super::SVF;
use crate::filter_parameters::FilterParameters;
// use crate::filter_parameters::FilterParameters::PluginParameters;

use crate::parameter::{ParameterF32, ParameterUsize};
use crate::vst::plugin::{PluginParameters, HostCallback};
use vst::host::Host;
mod plot;
use crate::utils::AtomicOps;
use vst::editor::Editor;
use baseview::{Size, WindowOpenOptions, WindowScalePolicy};

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;

const WINDOW_WIDTH: usize = 512;
const WINDOW_HEIGHT: usize = 512;
const WINDOW_WIDTH_F: f32 = WINDOW_WIDTH as f32;
const WINDOW_HEIGHT_F: f32 = WINDOW_HEIGHT as f32;
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const CYAN: [f32; 4] = [0.1, 1.0, 1.0, 0.8];
// const BG_COLOR: [f32; 4] = [0.21 * 1.4, 0.11 * 1.7, 0.25 * 1.4, 1.0];
// const BG_COLOR_TRANSP: [f32; 4] = [0.21 * 1.4, 0.11 * 1.7, 0.25 * 1.4, 0.0];
// const GREEN: [f32; 4] = [0.23, 0.68, 0.23, 1.0];
// const RED: [f32; 4] = [0.98, 0.02, 0.22, 1.0];
const ORANGE: [f32; 4] = [1.0, 0.58, 0.0, 1.0];
const ORANGE_HOVERED: [f32; 4] = [1.0, 0.68, 0.1, 1.0];
// const WAVEFORM_LINES: [f32; 4] = [1.0, 1.0, 1.0, 0.2];
const TEXT: [f32; 4] = [1.0, 1.0, 1.0, 0.75];
// const DB_LINES: [f32; 4] = [1.0, 1.0, 1.0, 0.15];

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







pub struct EditorState {
    pub params: Arc<FilterParameters>,
    pub host: Option<HostCallback>,
    // pub sample_rate: Arc<AtomicFloat>,
    // pub time: Arc<AtomicFloat>,
}
impl EditorState {
    fn draw_bode_plot(&self, ui: &Ui, size: [f32; 2]) {
        let draw_list = ui.get_window_draw_list();
        let cursor = ui.cursor_screen_pos();
        // draw a box with slightly different color?
        
        
        let color = ui.style_color(StyleColor::PlotLinesHovered);
        // let color = CYAN;
        let amps = plot::get_bode_array(self.params.g.get(), self.params.res.get(), self.params.mode.get());
        let length = amps.len();
        let scale = (size[0] as f32 / length as f32) as f32;
        let mut last = amps[0];
        let v_center = size[1] / 2.0;
        for i in 1..length {
            let next = amps[i];
            let fi = i as f32;
            //draw line from i to i+1? how to get start/endpoints?
            draw_list
            .add_line(
                [cursor[0] + fi * scale , cursor[1] + v_center - last],
                [cursor[0] + fi * scale  + 1., cursor[1] + v_center - next],
                color,
            ).thickness(5.).build();
            last = next;
        }


    }
    // Todo: Potentially we can avoid passing in parameter reference and just use parameter_index to get stuff we need
    pub fn make_knob(&self,
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
        let knob_id = &ImString::new(format!("##{}_KNOB_CONTORL_", title));
        knob_title(ui, &ImString::new(title.to_uppercase()), width);
        let cursor = ui.cursor_pos();
        ui.set_cursor_pos([cursor[0], cursor[1] + 5.0]);
        let mut val = parameter.get_normalized(); 
        let knob = Knob::new_custom_slope(
            ui,
            knob_id,
            &mut val,
            // 0.,
            // 1.,
            (parameter.get_func)(parameter.from_range(parameter.default)),
            width * 0.5,
            true,
            200.
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
    pub fn make_steppy_knob(&self,
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
        let knob = Knob::new_custom_slope(
            ui,
            knob_id,
            &mut val,
            // 0.,
            // 1.,
            (parameter.get_func)(parameter.from_range(parameter.default as f32)),
            width * 0.5,
            true,
            200.
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
        // TODO: Proper colors pls
        draw_stepped_knob(&knob, (parameter.max - parameter.min + 1.) as u32,  wiper_color, track_color, &ColorSet::from(CYAN));
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
                    let _graph_v_center = 225.0 + 25.0;
                    // {
                    //     let draw_list = ui.get_window_draw_list();
                    //     draw_list.add_rect_filled_multicolor(
                    //         [0.0, 0.0],
                    //         [WINDOW_WIDTH_F, 200.0],
                    //         BLACK,
                    //         BLACK,
                    //         BG_COLOR,
                    //         BG_COLOR,
                    //     );
                    //     draw_list
                    //         .add_rect([0.0, 200.0], [WINDOW_WIDTH_F, WINDOW_HEIGHT_F], BG_COLOR)
                    //         .filled(true)
                    //         .build();
                    //     draw_list
                    //         .add_rect(
                    //             [0.0, graph_v_center - 92.0],
                    //             [WINDOW_WIDTH_F, graph_v_center + 92.0],
                    //             [0.0, 0.0, 0.0, 0.65],
                    //         )
                    //         .filled(true)
                    //         .build();
                    // }
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

                    state.make_steppy_knob(ui, &params.mode, 3, &highlight, &lowlight, 0.0);
                    ui.next_column();

                    
                    move_cursor(ui, 0.0, 84.0);

                    ui.columns(1, im_str!("nocols"), false);
                    state.draw_bode_plot(ui, [400., 200.]);

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
