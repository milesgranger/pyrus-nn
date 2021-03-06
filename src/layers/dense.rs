use ndarray::Array2;
use ndarray_parallel::prelude::*;
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;
use serde_derive::{Deserialize, Serialize};

use crate::activations::{self, Activation};
use crate::layers::Layer;

#[derive(Default, Serialize, Deserialize)]
pub struct Dense {
    pub weights: Array2<f32>,
    pub n_input: usize,
    pub n_output: usize,
    pub output: Option<Array2<f32>>,
    pub input: Option<Array2<f32>>,
    pub activation: Activation,
}

#[typetag::serde]
impl Layer for Dense {
    fn new(n_input: usize, n_output: usize, activation: Activation) -> Self {
        let weights = Array2::<f32>::random((n_input, n_output), F32(Normal::new(-1., 1.)));
        Dense {
            weights,
            n_input,
            n_output,
            output: None,
            input: None,
            activation,
        }
    }
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.input = Some(x.clone());
        self.output = match self.activation {
            Activation::Linear => Some(x.dot(&self.weights)),
            Activation::Sigmoid => Some(activations::sigmoid(&x.dot(&self.weights), false)),
            Activation::Tanh => Some(activations::tanh(&x.dot(&self.weights), false)),
            Activation::Softmax => Some(activations::softmax(&x.dot(&self.weights), false)),
        };
        self.output.clone().unwrap()
    }
    fn n_input(&self) -> usize {
        self.n_input
    }
    fn n_output(&self) -> usize {
        self.n_output
    }
    fn output(&self) -> Array2<f32> {
        self.output.clone().unwrap()
    }
    fn input(&self) -> Array2<f32> {
        self.input.clone().unwrap()
    }
    fn weights(&self) -> Array2<f32> {
        self.weights.clone()
    }
    fn set_weights(&mut self, weights: Array2<f32>) -> () {
        self.weights = weights;
    }
    fn backward(&mut self, error: Array2<f32>, lr: f32) -> Array2<f32> {
        let delta = match self.activation {
            Activation::Sigmoid => activations::sigmoid(&self.output(), true) * error.t(),
            Activation::Linear => self.output() * error.t(),
            Activation::Tanh => activations::tanh(&self.output(), true) * error.t(),
            Activation::Softmax => activations::softmax(&self.output(), true) * error.t(),
        };
        let mut updates = self.input().t().dot(&delta);
        updates.par_mapv_inplace(|v| v * lr);
        let error_out = self.weights().dot(&delta.t());
        self.weights += &updates;
        error_out
    }
}
