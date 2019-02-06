use ndarray::Array2;

use crate::layers::Layer;
use crate::activations;

#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>
}

impl Sequential {
    pub fn new() -> Self {
        Sequential::default()
    }
    pub fn add(&mut self, layer: impl Layer + 'static) -> Result<(), &'static str> {

        // Ensure this layer's input matches the previous layer's output
        if self.len() > 0 {
            if self.layers[self.len() - 1].n_output() != layer.n_input() {
                return Err("Input shape mismatch!")  // TODO: Improve error msg.
            }
        }
        self.layers.push(Box::new(layer));
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn predict(&self, x: Array2<f32>) -> Array2<f32> {
        let mut layer0 = x;
        self.layers
            .iter()
            .fold(layer0, |out, layer| layer.dot(out))
    }
}