// colorhexa.com is nice for looking for colors

use crate::filter_parameters::FilterParameters;

// use crate::parameter::{ParameterF32, ParameterUsize};
use vst::plugin::{HostCallback};
mod plot;
use vst::editor::Editor;

use vizia::*;
use crate::ui::*;

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;

pub const WINDOW_WIDTH: u32 = 512;
pub const WINDOW_HEIGHT: u32 = 512;


/// keeps track of parameters and enables contact with the host
pub struct EditorState {
    pub params: Arc<FilterParameters>,
    pub host: Option<HostCallback>,
}
impl EditorState {
    pub fn new(params: Arc<FilterParameters>, host: Option<HostCallback>) -> EditorState {
        EditorState {
            params,
            host,
        }
    }
    // fn draw_bode_plot(&self, ui: &Ui, size: [f32; 2]) {
    //     let draw_list = ui.get_window_draw_list();
    //     let cursor = ui.cursor_screen_pos();
    //     // adding a background
    //     draw_list
    //         .add_rect(
    //             [cursor[0], cursor[1] - size[1]],
    //             [cursor[0] + size[0], cursor[1]],
    //             GREY,
    //         )
    //         .filled(true)
    //         .thickness(5.)
    //         .build();

    //     let color = ORANGE;
    //     let mut amps: Vec<f32>;
    //     if self.params.filter_type.get() == 0 {
    //         amps = plot::get_filter_bode(
    //             self.params.cutoff.get(),
    //             self.params.zeta.get(),
    //             self.params.mode.get(),
    //             self.params.filter_type.get(),
    //         );
    //     } else {
    //         amps = plot::get_filter_bode(
    //             self.params.cutoff.get(),
    //             self.params.k_ladder.get(),
    //             self.params.slope.get(),
    //             self.params.filter_type.get(),
    //         );
    //     };

    //     let maxmin = 40.;
    //     // normalizing amplitudes
    //     for x in &mut amps {
    //         *x = (*x - (-maxmin)) / (maxmin - (-maxmin))
    //     }
    //     let length = amps.len();
    //     let scale = (size[0] / length as f32) as f32;
    //     // let scale_y = size[1] / 2.;
    //     let scale_y = size[1];
    //     let mut last = amps[0] * scale_y;
    //     for i in 1..length {
    //         // The scale might give problems with clipping out if resonance is higher than +12 dB
    //         let next = amps[i] * scale_y;

    //         let fi = i as f32;
    //         // only draw values that are within bounds
    //         if last > 0. && next < scale_y {
    //             //draw line from i to i+1
    //             draw_list
    //                 .add_line(
    //                     [cursor[0] + fi * scale, cursor[1] - last],
    //                     [cursor[0] + fi * scale + 1., cursor[1] - next],
    //                     color,
    //                 )
    //                 .thickness(5.)
    //                 .build();
    //         }

    //         last = next;
    //     }
    //     // adding a frame that covers up some weird stuff with end lines
    //     draw_list
    //         .add_rect(
    //             [cursor[0], cursor[1] - size[1]],
    //             [cursor[0] + size[0], cursor[1]],
    //             BLACK,
    //         )
    //         .filled(false)
    //         .thickness(8.)
    //         .build();
    // }
    // Todo: Potentially we can avoid passing in parameter reference and just use parameter_index to get stuff we need
  
}
pub struct SVFPluginEditor {
    pub is_open: bool,
    pub state: Arc<EditorState>,
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

        let state = self.state.clone();
        let window_description = WindowDescription::new()
            .with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
            .with_title("Hello Plugin");

        Application::new(window_description, move |cx| {

            plugin_gui(cx, state.clone());

        }).open_parented(&ParentWindow(parent));

        // self.handle = Some(window_handle);
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
