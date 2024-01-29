use godot::prelude::*;

struct GMindExtension;

#[gdextension]
unsafe impl ExtensionLibrary for GMindExtension {
    
}
mod dense;
