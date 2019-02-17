use ndarray::{Array1, Array2};
use ndarray_parallel::prelude::*;

use crate::layers::Layer;
use crate::activations;

#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    lr: f32,
    n_epoch: usize
}

fn squared_error(y: f32, yhat: f32) -> f32 {
    (y - yhat).abs().sqrt()
}

impl Sequential {

    /// Create a new `Sequential` network.
    pub fn new() -> Self {
        let mut nn = Sequential::default();
        nn.lr = 0.1;
        nn.n_epoch = 100;
        nn
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
    
    /// Apply the network against an input
    pub fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.layers
            .iter_mut()
            .fold(x, |out, ref mut layer| layer.forward(out))
    }

    /// Run back propagation on output vs expected
    pub fn backward(&mut self, output: &Array2<f32>, expected: &Array2<f32>) {
        self.layers
            .iter_mut()
            .rev()
            .fold(None, | error: Option<Array2<f32>>, layer: &mut Box<dyn Layer + 'static> | {

                match error {

                    // All hidden and input layers
                    Some(error) => {
                        let error_out = layer.backward(error);
                        Some(error_out)
                    },

                    // Output layer, (no error calculated from previous layer)
                    None => {
                        // TODO: Add choice of cost func.
                        let mut error = expected - output;
                        error.par_mapv_inplace(|v| v.abs().sqrt());

                        let error_out = layer.backward(error.t().to_owned());
                        Some(error_out)
                    }
                }

            });
    }

    /// Train the network according to the parameters set given training and target data
    pub fn fit(&mut self, x: Array2<f32>, y: Array2<f32>) {

        for epoch in 0..self.n_epoch {
            let output = self.forward(x.clone());
            self.backward(&output, &y)
        }

    }

}