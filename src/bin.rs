#![feature(portable_simd)]
use std::sync::Arc;

use vizia::{Application, WindowDescription};

// mod editor;
// use editor::{EditorState, WINDOW_HEIGHT, WINDOW_WIDTH};
mod filter_params_nih;
mod parameter;
#[allow(dead_code)]
mod utils;
use filter_params_nih::FilterParams;

mod filter;

mod ui;
use ui::*;

// fn main() {
//     let params = Arc::new(FilterParams::new());
//     // let state = Arc::new(EditorState::new(params.clone(), None));

//     let window_description = WindowDescription::new()
//         .with_inner_size(0, 0)
//         .with_title("Hello Plugin");

//     Application::new(window_description, move |cx| {
//         cx.add_stylesheet("src/style.css")
//             .expect("no style sheet found");

//         // plugin_gui(cx, Arc::clone(&params));
//         plugin_gui(cx, Arc::clone(&params));
//     })
//     .run();
// }
