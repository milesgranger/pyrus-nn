use ndarray::{Array2, Axis};
use ndarray_parallel::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Linear,
    Tanh,
    Softmax,
}

impl From<String> for Activation {
    fn from(name: String) -> Self {
        match name.to_lowercase().as_str() {
            "sigmoid" => Activation::Sigmoid,
            "linear" => Activation::Linear,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            _ => panic!("Activation {} not supported", name)
        }
    }
}

impl std::default::Default for Activation {
    fn default() -> Self {
        Activation::Linear
    }
}

/// Softmax
pub fn softmax(x: &Array2<f32>, deriv: bool) -> Array2<f32> {
    let mut out = x.clone();
    let _ = out
        .axis_iter_mut(Axis(0))
        .map(|ref mut vec| {
            let max = vec
                .iter()
                .max_by(|a, b| a.partial_cmp(&b).unwrap_or_else(|| Ordering::Equal))
                .unwrap();
            let exps = vec.mapv(|v| (v - max).exp());
            let result = &exps / exps.sum();
            vec.zip_mut_with(&result, |v, r| *v = *r);

            // Derivative
            if deriv {
                let _shape = vec.shape();
                let s = vec.to_owned();
                let result = s.diag().to_owned() - s.dot(&s.t());
                vec.zip_mut_with(&result, |v, r| *v = *r);
            }
        })
        .collect::<Vec<()>>();
    out
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

fn _tanh(x: f32) -> f32 {
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
