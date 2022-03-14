#![feature(portable_simd)]
use std::{pin::Pin, sync::Arc};

use nih_plug::context::GuiContext;
use vizia::{Application, WindowDescription};

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
    let window_description = WindowDescription::new()
        .with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .with_title("Hello Plugin");

    Application::new(window_description, move |cx| {
        cx.add_stylesheet("src/style.css")
            .expect("no style sheet found");

        // plugin_gui(cx, Arc::clone(&params));
        plugin_gui(cx, Pin::new(Arc::clone(&params)), param_set_guy.clone());
    })
    .run();
}
// dummmy GuiContext for the standalone version
struct ParamSetGuy();

impl GuiContext for ParamSetGuy {
    unsafe fn raw_begin_set_parameter(&self, _param: nih_plug::param::internals::ParamPtr) {
        unimplemented!()
    }

    unsafe fn raw_set_parameter_normalized(
        &self,
        param: nih_plug::param::internals::ParamPtr,
        normalized: f32,
    ) {
        param.set_normalized_value(normalized);
    }

    unsafe fn raw_end_set_parameter(&self, _param: nih_plug::param::internals::ParamPtr) {
        unimplemented!()
    }

    unsafe fn raw_default_normalized_param_value(
        &self,
        _param: nih_plug::param::internals::ParamPtr,
    ) -> f32 {
        todo!()
    }
}
