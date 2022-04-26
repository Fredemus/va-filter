#![feature(portable_simd)]
use std::sync::Arc;

use nih_plug::context::GuiContext;
use vizia::Application;

mod editor;
use editor::{WINDOW_HEIGHT, WINDOW_WIDTH};
mod filter_params_nih;
mod parameter;
#[allow(dead_code)]
mod utils;
use filter_params_nih::FilterParams;

mod filter;

mod ui;
use ui::*;

fn main() {
    let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = Arc::new(FilterParams::new(should_update_filter));
    // let state = Arc::new(EditorState::new(params.clone(), None));
    let param_set_guy = Arc::new(ParamSetGuy {});

    Application::new(move |cx| {
        cx.add_stylesheet("src/style.css")
            .expect("no style sheet found");

        // plugin_gui(cx, Arc::clone(&params));
        plugin_gui(cx, Arc::clone(&params), param_set_guy.clone());
    })
    .inner_size((WINDOW_WIDTH, WINDOW_HEIGHT))
    .title("Hello Plugin")
    .run();
}
// dummmy GuiContext for the standalone version
struct ParamSetGuy();

impl GuiContext for ParamSetGuy {
    unsafe fn raw_begin_set_parameter(&self, _param: nih_plug::param::internals::ParamPtr) {}

    unsafe fn raw_set_parameter_normalized(
        &self,
        param: nih_plug::param::internals::ParamPtr,
        normalized: f32,
    ) {
        param.set_normalized_value(normalized);
    }

    unsafe fn raw_end_set_parameter(&self, _param: nih_plug::param::internals::ParamPtr) {}

    fn request_resize(&self) -> bool {
        todo!()
    }

    fn get_state(&self) -> nih_plug::prelude::PluginState {
        todo!()
    }

    fn set_state(&self, _state: nih_plug::prelude::PluginState) {
        todo!()
    }

    fn plugin_api(&self) -> nih_plug::context::PluginApi {
        todo!()
    }
}
