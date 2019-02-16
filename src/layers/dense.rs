use ndarray::{ArrayD, Axis, Dim, Array2, Array1};
use rand::distributions::Normal;
use ndarray_rand::{RandomExt, F32};

use crate::layers::Layer;
use crate::activations;

#[derive(Default)]
pub struct Dense {
    pub weights: Array2<f32>,
    pub n_input: usize,
    pub n_output: usize,
    pub output: Option<Array2<f32>>,
    pub input: Option<Array2<f32>>
}

impl Layer for Dense {

    fn new(n_input: usize, n_output: usize) -> Self {

        let weights = Array2::<f32>::random(
            (n_input, n_output),
            F32(Normal::new(-1., 1.))
        );
        Dense { weights, n_input, n_output, output: None, input: None }
    }
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.input = Some(x.clone());
        self.output = Some(activations::sigmoid(&x.dot(&self.weights), false));
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
    fn backward(&mut self, updates: Array2<f32>) {
        self.weights.zip_mut_with(&updates, |v1, v2| *v1 += v2);
    }
}
