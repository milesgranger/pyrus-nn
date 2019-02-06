use ndarray::{ArrayD, Dim, Array2};
use rand::distributions::Normal;
use ndarray_rand::{RandomExt, F32};

use crate::layers::Layer;

#[derive(Default)]
pub struct Dense {
    weights: Array2<f32>,
    n_input: usize,
    n_output: usize,
}

impl Layer for Dense {

    fn new(n_input: usize, n_output: usize) -> Self {

        let weights = Array2::<f32>::random(
            (n_output, n_input),
            F32(Normal::new(0., 1.))
        );
        Dense { weights, n_input, n_output }
    }
    fn dot(&self, x: Array2<f32>) -> Array2<f32> {
        self.weights.dot(&x.t())
    }
    fn n_input(&self) -> usize {
        self.n_input
    }
    fn n_output(&self) -> usize {
        self.n_output
    }

}
