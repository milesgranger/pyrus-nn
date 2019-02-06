use ndarray::Array2;

mod dense;

// Exports
pub use crate::layers::dense::Dense;


pub trait Layer {
    fn new(n_input: usize, n_output: usize) -> Self where Self: Sized;
    fn dot(&self, x: Array2<f32>) -> Array2<f32>;
    fn n_input(&self) -> usize;
    fn n_output(&self) -> usize;
}