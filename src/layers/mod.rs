use ndarray::Array2;

mod dense;

// Exports
pub use crate::activations::Activation;
pub use crate::layers::dense::Dense;

#[typetag::serde]
pub trait Layer {
    fn new(n_input: usize, n_output: usize, activation: Activation) -> Self
    where
        Self: Sized;
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32>;
    fn n_input(&self) -> usize;
    fn n_output(&self) -> usize;
    fn output(&self) -> Array2<f32>;
    fn input(&self) -> Array2<f32>;
    fn weights(&self) -> Array2<f32>;
    fn backward(&mut self, error: Array2<f32>, lr: f32) -> Array2<f32>;
}
