use std::iter;

use ndarray::{Array2, ArrayView2, Axis};
use rand::{self, Rng};
use serde_derive::{Deserialize, Serialize};

use crate::costs::{self, CostFunc};
use crate::layers::Layer;

#[derive(Default, Serialize, Deserialize)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    pub lr: f32,
    pub n_epoch: usize,
    pub batch_size: usize,
    pub cost: CostFunc,
    pub verbose: bool,
}

impl Sequential {
    /// Create a new `Sequential` network, with _perhaps_ sensible defaults.
    pub fn new(lr: f32, n_epoch: usize, batch_size: usize, cost: CostFunc) -> Self {
        Sequential {
            layers: vec![],
            lr,
            n_epoch,
            batch_size,
            cost,
            verbose: true,
        }
    }

    /// Add a layer to the network
    pub fn add(&mut self, layer: impl Layer + 'static) -> Result<(), &'static str> {
        // Ensure this layer's input matches the previous layer's output
        if self.len() > 0 {
            if self.layers[self.len() - 1].n_output() != layer.n_input() {
                return Err("Input shape mismatch!"); // TODO: Improve error msg.
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
        let lr = self.lr;

        self.layers.iter_mut().rev().fold(
            None,
            |error: Option<Array2<f32>>, layer: &mut Box<dyn Layer + 'static>| {
                match error {
                    // All hidden and input layers
                    Some(error) => {
                        let error_out = layer.backward(error, lr);
                        Some(error_out)
                    }

                    // Output layer, (no error calculated from previous layer)
                    None => {
                        let error = &expected - &output;

                        let error_out = layer.backward(error.t().to_owned(), lr);
                        Some(error_out)
                    }
                }
            },
        );
    }

    /// Train the network according to the parameters set given training and target data
    pub fn fit(&mut self, x: ArrayView2<f32>, y: ArrayView2<f32>) {
        let x_len = x.shape()[0];

        // Epochs
        for epoch in 1..self.n_epoch + 1 {
            // Next shuffle index
            let mut rng = rand::thread_rng();
            let index = (0..x_len)
                .map(|_| rng.gen_range(0, x_len))
                .collect::<Vec<usize>>();

            for chunk_slice_idx in index.as_slice().chunks(self.batch_size) {
                // TODO: Find a way not to create new array but array view
                let batch = x.select(Axis(0), chunk_slice_idx);
                let target = y.select(Axis(0), chunk_slice_idx);

                let output = self.forward(batch.view());
                self.backward(output.view(), target.view())
            }

            // Output some stats
            if self.verbose {
                let output = self.forward(x.view());

                let score = match self.cost {
                    CostFunc::MSE => costs::mean_squared_error(y.view(), output.view()),
                    CostFunc::MAE => costs::mean_absolute_error(y.view(), output.view()),
                    CostFunc::Accuracy => costs::accuracy_score(y.view(), output.view()),
                    CostFunc::CrossEntropy => costs::cross_entropy(y.view(), output.view()),
                };

                let progress = ((epoch as f32 / self.n_epoch as f32) * 10.) as usize;
                let bar = iter::repeat("=").take(progress).collect::<String>();
                let space_left = iter::repeat(".").take(10 - progress).collect::<String>();
                println!(
                    "{}",
                    format!(
                        "[{}>{}] - Epoch: {} - Error: {:.4}",
                        bar, space_left, epoch, score
                    )
                );
            }
        }
    }

    /// Mutably iterate over the layers in the network.
    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut Box<dyn Layer>> {
        self.layers.iter_mut()
    }

    /// Iterate over the layers in the network.
    pub fn iter(&self) -> impl Iterator<Item=&Box<dyn Layer>> {
        self.layers.iter()
    }
}
