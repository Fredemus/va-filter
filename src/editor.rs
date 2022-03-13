use baseview::WindowHandle;
use nih_plug::{context::GuiContext, plugin::{Editor, ParentWindowHandle}};
use vizia::{WindowDescription, Application};
use std::sync::Arc;

pub use vizia::*;

const STYLE: &str = include_str!("style.css");

pub const WINDOW_WIDTH: u32 = 512;
pub const WINDOW_HEIGHT: u32 = 512;


pub fn create_vizia_editor<U>(
    update: U,
) -> Option<Box<dyn Editor>>
where
    U: Fn(&mut Context, Arc<dyn GuiContext>) + 'static + Send + Sync,
{
    Some(Box::new(ViziaEditor {
        update: Arc::new(update),
    }))
}

pub struct ViziaEditor {
    update: Arc<dyn Fn(&mut Context, Arc<dyn GuiContext>) + 'static + Send + Sync>,
}

impl Editor for ViziaEditor {
    fn spawn(&self, parent: ParentWindowHandle, context: Arc<dyn GuiContext>) -> Box<dyn std::any::Any + Send + Sync> {
        let update = self.update.clone();

        let window_description = WindowDescription::new().with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT);
        let window = Application::new(window_description, move |cx|{
            cx.add_theme(STYLE);

            (update)(cx, context.clone());
        }).open_parented(&parent);

        Box::new(ViziaEditorHandle{
            window,
        })
    }

    fn size(&self) -> (u32, u32) {
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    }

    fn set_scale_factor(&self, _factor: f32) -> bool {
        todo!()
    }
}

struct ViziaEditorHandle {
    window: WindowHandle,
}

unsafe impl Send for ViziaEditorHandle {}
unsafe impl Sync for ViziaEditorHandle {}

impl Drop for ViziaEditorHandle {
    fn drop(&mut self) {
        self.window.close();
    }
}

// /// keeps track of parameters and enables contact with the host
// pub struct EditorState {
//     pub params: Arc<FilterParameters>,
//     pub host: Option<HostCallback>,
// }
// impl EditorState {
//     pub fn new(params: Arc<FilterParameters>, host: Option<HostCallback>) -> EditorState {
//         EditorState { params, host }
//     }
// }
// pub struct SVFPluginEditor {
//     pub is_open: bool,
//     pub state: Arc<EditorState>,
//     pub handle: Option<WindowHandle>,
// }

// impl Editor for SVFPluginEditor {
//     fn position(&self) -> (i32, i32) {
//         (0, 0)
//     }

//     fn size(&self) -> (i32, i32) {
//         (WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32)
//     }

//     fn open(&mut self, parent: *mut ::std::ffi::c_void) -> bool {
//         if self.is_open {
//             return false;
//         }

//         self.is_open = true;

//         let state = self.state.clone();
//         let window_description = WindowDescription::new()
//             .with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
//             .with_title("SVF");

//         let handle = Application::new(window_description, move |cx| {
//             cx.add_theme(STYLE);

//             plugin_gui(cx, state.clone());
//         })
//         .open_parented(&ParentWindow(parent));
//         self.handle = Some(handle);

//         true
//     }

//     fn is_open(&mut self) -> bool {
//         self.is_open
//     }

//     fn close(&mut self) {
//         self.is_open = false;
//         if let Some(mut handle) = self.handle.take() {
//             handle.close();
//         }
//     }
// }
// struct VstParent(*mut ::std::ffi::c_void);

// #[cfg(target_os = "macos")]
// unsafe impl HasRawWindowHandle for VstParent {
//     fn raw_window_handle(&self) -> RawWindowHandle {
//         use raw_window_handle::macos::MacOSHandle;

//         RawWindowHandle::MacOS(MacOSHandle {
//             ns_view: self.0 as *mut ::std::ffi::c_void,
//             ..MacOSHandle::empty()
//         })
//     }
// }

// #[cfg(target_os = "windows")]
// unsafe impl HasRawWindowHandle for VstParent {
//     fn raw_window_handle(&self) -> RawWindowHandle {
//         use raw_window_handle::windows::WindowsHandle;

//         RawWindowHandle::Windows(WindowsHandle {
//             hwnd: self.0,
//             ..WindowsHandle::empty()
//         })
//     }
// }

// #[cfg(target_os = "linux")]
// unsafe impl HasRawWindowHandle for VstParent {
//     fn raw_window_handle(&self) -> RawWindowHandle {
//         use raw_window_handle::unix::XcbHandle;

//         RawWindowHandle::Xcb(XcbHandle {
//             window: self.0 as u32,
//             ..XcbHandle::empty()
//         })
//     }
// }
