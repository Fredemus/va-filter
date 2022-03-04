#![feature(portable_simd)]
use std::sync::Arc;

use vizia::{Application, WindowDescription};

mod editor;
use editor::{EditorState, WINDOW_HEIGHT, WINDOW_WIDTH};
mod filter_parameters;
mod parameter;
#[allow(dead_code)]
mod utils;
use filter_parameters::FilterParameters;

mod filter;

mod ui;
use ui::*;

fn main() {
    let params = Arc::new(FilterParameters::default());
    let state = Arc::new(EditorState::new(params.clone(), None));
    let mut window_description = WindowDescription::new()
        .with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .with_title("Hello Plugin");
    window_description.resizable = false;

    Application::new(window_description, move |cx| {
        cx.add_stylesheet("src/style.css")
            .expect("no style sheet found");

        // plugin_gui(cx, Arc::clone(&params));
        plugin_gui(cx, Arc::clone(&state));
    })
    .run();
}
