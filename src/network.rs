use std::iter;

use ndarray::{Array1, Array2, ArrayBase, Zip, ArrayView2};
use ndarray_parallel::prelude::*;

use crate::layers::Layer;
use crate::activations;
use crate::costs::{self, CostFunc};

#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    lr: f32,
    n_epoch: usize,
    batch_size: usize,
    cost: CostFunc,
    verbose: bool,
}


impl Sequential {

    /// Create a new `Sequential` network, with _perhaps_ sensible defaults.
    pub fn new() -> Self {
        let mut nn = Sequential::default();
        nn.lr = 0.1;
        nn.n_epoch = 10;
        nn.batch_size = 32;
        nn.verbose = true;
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
    pub fn predict(&mut self, x: ArrayView2<f32>) -> Array2<f32> {
        self.forward(x)
    }
    
    /// Apply the network against an input
    pub fn forward(&mut self, x: ArrayView2<f32>) -> Array2<f32> {
        self.layers
            .iter_mut()
            .fold(x.to_owned(), |out, ref mut layer| layer.forward(out))
    }

    /// Run back propagation on output vs expected
    pub fn backward(&mut self, output: ArrayView2<f32>, expected: ArrayView2<f32>) {

        let cost_func = match self.cost {
            CostFunc::MSE => costs::squared_error
        };

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

                        // Hold element-wise errors
                        let mut error = Array2::zeros((output.shape()[0], output.shape()[1]));

                        // Apply the scalar based cost function to out vs expected elements
                        Zip::from(&mut error)
                            .and(&expected)
                            .and(&output)
                            .par_apply(|err, &exp, &out| {
                                *err = cost_func(exp, out);
                            });

                        let error_out = layer.backward(error.t().to_owned());
                        Some(error_out)
                    }
                }

            });
    }

    /// Train the network according to the parameters set given training and target data
    pub fn fit(&mut self, x: ArrayView2<f32>, y: ArrayView2<f32>) {

        // Epochs
        for epoch in 1..self.n_epoch + 1 {

            // Batches
            for (batch, target) in x.exact_chunks((self.batch_size, x.shape()[1]))
                .into_iter().zip(y.exact_chunks((self.batch_size, y.shape()[1])).into_iter()) {

                let output = self.forward(batch);
                self.backward(output.view(), target.view());
            }

            // Output some stats
            if self.verbose {

                let output = self.forward(x.view());

                let error = match self.cost {
                    CostFunc::MSE => costs::mean_squared_error(y.view(), output.view())
                };

                let progress = ((epoch as f32 / self.n_epoch as f32) * 10.) as usize;
                let bar = iter::repeat("=").take(progress).collect::<String>();
                let space_left = iter::repeat(".").take(10 - progress).collect::<String>();
                println!("{}", format!("[{}>{}] - Epoch: {} - Error: {:.4}", bar, space_left, epoch, error));

            }
        }

    }

}