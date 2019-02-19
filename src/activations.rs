use std::ops::Mul;
use ndarray::Array2;
use ndarray_parallel::prelude::*;


pub enum Activation {
    Sigmoid,
    Linear,
    Tanh
}

impl std::default::Default for Activation {
    fn default() -> Self { Activation::Linear }
}

/// Tanh
pub fn tanh(x: &Array2<f32>, deriv: bool) -> Array2<f32> {
    let mut out = x.clone();

    if deriv {
        out.par_mapv_inplace(_tanh_prime);
    } else {
        out.par_mapv_inplace(_tanh);
    }
    out
}

fn _tanh(x:f32) -> f32 {
    let y = (-2.0 * x).exp();
    (1.0 - y) / (1.0 + y)
}
fn _tanh_prime(x: f32) -> f32 {
    _tanh(x) * (1.0 - _tanh(x))
}

/// Calculate result of a sigmoid on `ArrayD`
pub fn sigmoid(x: &Array2<f32>, deriv: bool) -> Array2<f32> {
    // TODO: Make this an inplace (&mut) process
    let mut out = x.clone();

    if deriv {
        out.par_mapv_inplace(_sigmoid_prime);
    } else {
        out.par_mapv_inplace(_sigmoid);
    }
    out
}

fn _sigmoid_prime(x: f32) -> f32 {
    _sigmoid(x) * (1. - _sigmoid(x))
}
fn _sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
