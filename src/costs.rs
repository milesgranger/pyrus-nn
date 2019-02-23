use std::f32::MIN;
use ndarray::{ArrayView2, Axis};

/// Cost function selection `enum`
pub enum CostFunc {
    MSE,
    MAE,
    Accuracy,
    CrossEntropy
}

impl std::default::Default for CostFunc {
    fn default() -> Self { CostFunc::MSE }
}

/// Cross entropy; aka logistic loss / log loss
pub fn cross_entropy(y_true: ArrayView2<f32>, y_hat: ArrayView2<f32>) -> f32 {

    let elipson = 1e-15;  // TODO: Make this a parameter

    // Clip
    let mut y_hat = y_hat.mapv(|v| if v > 1.0 { v - elipson } else { v + elipson }).to_owned();

    // Normalize
    y_hat = y_hat.to_owned() / y_hat.sum_axis(Axis(1)).into_shape((y_hat.shape()[0], 1)).unwrap().to_owned();

    // Loss
    -(y_true.to_owned() * y_hat.mapv(|v| v.ln())).sum_axis(Axis(1)).sum() / y_hat.rows() as f32
}

/// Cross entropy score of single element
pub fn single_cross_entropy(y_true: f32, y_hat: f32) -> f32 {

    let y_hat = if y_hat > 1.0 { 1.0 - 1e-15 } else if y_hat < 0.0 { 0.0 + 1e-15 } else { y_hat };
    -(y_hat.ln() + ( 1. - y_true) * (1. - y_hat).ln())
}

/// Measure accuracy score
pub fn accuracy_score(y_true: ArrayView2<f32>, y_hat: ArrayView2<f32>) -> f32 {
    y_true.outer_iter()
        .zip(y_hat.outer_iter())
        .map(|(yt, yh)| {
            if yt.len() > 1 {
                let (ytrue_argmax, _max) = yt.iter()
                    .enumerate()
                    .fold((None, MIN), |(idx, acc), (i, x)| {
                        if x > &acc {
                            (Some(i), *x)
                        } else {
                            (idx, acc)
                        }
                    });

                let (yhat_argmax, _max) = yh.iter()
                    .enumerate()
                    .fold((None, MIN), |(idx, acc), (i, x)| {
                        if x > &acc {
                            (Some(i), *x)
                        } else {
                            (idx, acc)
                        }
                    });

                accuracy(ytrue_argmax.unwrap_or(0) as f32, yhat_argmax.unwrap_or(0) as f32)
            } else {
                accuracy(yt[0], yh[0])
            }

        })
        .sum::<f32>() / y_true.rows() as f32
}

/// Measure if two `f32` elements are equal; here for consistency.
pub fn accuracy(y_true: f32, y_hat: f32) -> f32 {
    if y_hat == y_true {
        1.
    } else {
        0.
    }
}

/// Determine the Mean Squared Error
pub fn mean_squared_error(y_true: ArrayView2<f32>, y_hat: ArrayView2<f32>) -> f32 {
    y_true.iter()
        .zip(y_hat.iter())
        .map(|(yt, yh)| squared_error(*yt, *yh))
        .sum::<f32>() / y_true.rows() as f32
}

/// Squared error between two `f32` values
pub fn squared_error(y_true: f32, y_hat: f32) -> f32 {
    (y_true - y_hat).powf(2.0)
}

/// Determin the Mean Absolute Error
pub fn mean_absolute_error(y_true: ArrayView2<f32>, y_hat: ArrayView2<f32>) -> f32 {
    y_true.iter()
        .zip(y_hat.iter())
        .map(|(yt, yh)| absolute_error(*yt, *yh))
        .sum::<f32>() / y_true.rows() as f32
}

/// Absolute error between two `f32` values
pub fn absolute_error(y_true: f32, y_hat: f32) -> f32 {
    (y_true - y_hat).abs()
}
