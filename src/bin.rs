use std::sync::Arc;

use vizia::{WindowDescription, Application};

mod editor;
use editor::{EditorState, WINDOW_WIDTH, WINDOW_HEIGHT};
mod parameter;
#[allow(dead_code)]
mod utils;
mod filter_parameters;
use filter_parameters::FilterParameters;

mod filter;


mod ui;
use ui::*;

fn main() {
    let params = Arc::new(FilterParameters::default());
    let state = Arc::new(EditorState::new(params.clone(), None));
    let window_description = WindowDescription::new()
        .with_inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .with_title("Hello Plugin");

    Application::new(window_description, move |cx|{

        // plugin_gui(cx, Arc::clone(&params));
        plugin_gui(cx, Arc::clone(&state));

    }).run();
}