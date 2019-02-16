use std::ops::Mul;
use ndarray::Array2;


/// Calculate result of a sigmoid on `ArrayD`
pub fn sigmoid(x: &Array2<f32>, deriv: bool) -> Array2<f32> {
    if deriv {
        x.mapv(_sigmoid_prime)
    } else {
        x.mapv(_sigmoid)
    }
}

fn _sigmoid_prime(x: f32) -> f32 {
    _sigmoid(x) * (1. - _sigmoid(x))
}

fn _sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
