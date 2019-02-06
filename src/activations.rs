use std::ops::Mul;
use std::f32::consts::E as Eurlers;
use ndarray::Array2;


/// Calculate result of a sigmoid on `ArrayD`
pub fn sigmoid(x: &Array2<f32>, deriv: bool) -> Array2<f32> {
    if deriv {
        x.mapv(|v| v*(1.-v))
    } else {
        1. / (1. + -x.mapv(|v| v.powf(Eurlers)))
    }
}
