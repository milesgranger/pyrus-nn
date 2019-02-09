use ndarray::{Array1, Array2};

use crate::layers::Layer;
use crate::activations;

#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>
}

impl Sequential {

    /// Create a new `Sequential` network.
    pub fn new() -> Self {
        Sequential::default()
    }

    /// Add a layer to the network
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

    /// Determine how many layers on held in the network
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Use the network to predict the outcome of x
    pub fn predict(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.forward(x)
    }
    
    // Apply the network against an input
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.layers
            .iter_mut()
            .fold(x, |out, ref mut layer| layer.forward(out))
    }

    // train network
    pub fn fit(&mut self, x: Array2<f32>, y: Array2<f32>) {

        let output = self.forward(x.clone());

        self.layers
            .iter_mut()
            .rev()
            .fold(None, | error: Option<Array2<f32>>, layer: &mut Box<dyn Layer + 'static>| {

                match error {

                    // All hidden and input layers
                    Some(error) => {
                        let delta_i = activations::sigmoid(&layer.output(), true) * error.t();
                        let error_out = layer.weights().dot(&delta_i.t());


                        let updates = layer.input().t().dot(&delta_i);
                        layer.backward(updates);

                        Some(error_out)
                    },

                    // Output layer, (no error calculated from previous layer)
                    None => {
                        let error = &y - &output;
                        let delta_o = error * activations::sigmoid(&output, true);
                        let error_out = layer.weights().dot(&delta_o.t());

                        let updates = layer.input().t().dot(&delta_o);
                        println!("Calcualted updates for output layer: {:?}", updates.shape());
                        layer.backward(updates);

                        Some(error_out)
                    }
                }

            });
    }

}