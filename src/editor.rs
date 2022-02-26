// colorhexa.com is nice for looking for colors

use crate::filter_parameters::FilterParameters;

use baseview::WindowHandle;
// use crate::parameter::{ParameterF32, ParameterUsize};
use vst::plugin::HostCallback;
mod plot;
pub use plot::*;
use vst::editor::Editor;

use crate::ui::*;
use vizia::*;

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;

const STYLE: &str = include_str!("style.css");

pub const WINDOW_WIDTH: u32 = 512;
pub const WINDOW_HEIGHT: u32 = 512;

/// keeps track of parameters and enables contact with the host
pub struct EditorState {
    pub params: Arc<FilterParameters>,
    pub host: Option<HostCallback>,
}
impl EditorState {
    pub fn new(params: Arc<FilterParameters>, host: Option<HostCallback>) -> EditorState {
        EditorState { params, host }
    }
}
pub struct SVFPluginEditor {
    pub is_open: bool,
    pub state: Arc<EditorState>,
    pub handle: Option<WindowHandle>,
}

impl Editor for SVFPluginEditor {
    fn position(&self) -> (i32, i32) {
        (0, 0)
    }

    fn size(&self) -> (i32, i32) {
        (WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32)
    }

    fn open(&mut self, parent: *mut ::std::ffi::c_void) -> bool {
        if self.is_open {
            return false;
        }

        self.is_open = true;

        let state = self.state.clone();
        let window_description = WindowDescription::new()
            .with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
            .with_title("SVF");

        let handle = Application::new(window_description, move |cx| {
            cx.add_theme(STYLE);

            plugin_gui(cx, state.clone());
        })
        .open_parented(&ParentWindow(parent));
        self.handle = Some(handle);

        true
    }

    fn is_open(&mut self) -> bool {
        self.is_open
    }

    fn close(&mut self) {
        self.is_open = false;
        if let Some(mut handle) = self.handle.take() {
            handle.close();
        }
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
