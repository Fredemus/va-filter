use baseview::WindowHandle;
use nih_plug::{
    context::GuiContext,
    plugin::{Editor, ParentWindowHandle},
};
use std::sync::Arc;
use vizia::Application;

pub use vizia::*;

const STYLE: &str = include_str!("style.css");

pub const WINDOW_WIDTH: u32 = 512;
pub const WINDOW_HEIGHT: u32 = 512;

pub fn create_vizia_editor<U>(update: U) -> Option<Box<dyn Editor>>
where
    U: Fn(&mut prelude::Context, Arc<dyn GuiContext>) + 'static + Send + Sync,
{
    Some(Box::new(ViziaEditor {
        update: Arc::new(update),
    }))
}

pub struct ViziaEditor {
    update: Arc<dyn Fn(&mut prelude::Context, Arc<dyn GuiContext>) + 'static + Send + Sync>,
}

impl Editor for ViziaEditor {
    fn spawn(
        &self,
        parent: ParentWindowHandle,
        context: Arc<dyn GuiContext>,
    ) -> Box<dyn std::any::Any + Send + Sync> {
        let update = self.update.clone();

        let window = Application::new(move |cx| {
            cx.add_theme(STYLE);

            (update)(cx, context.clone());
        })
        .inner_size((WINDOW_WIDTH, WINDOW_HEIGHT))
        .title("Hello Plugin")
        .open_parented(&parent);

        Box::new(ViziaEditorHandle { window })
    }

    fn size(&self) -> (u32, u32) {
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    }

    fn set_scale_factor(&self, _factor: f32) -> bool {
        true
        // todo!()
    }

    fn param_values_changed(&self) {
        // todo!()
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
