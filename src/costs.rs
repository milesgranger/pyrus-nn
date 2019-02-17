use ndarray::{ArrayView2, Zip};


/// Determine the Mean Squared Error between `expected` and `output`
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

/// Cost function selection `enum`
pub enum CostFunc {
    MSE,
}

impl std::default::Default for CostFunc {
    fn default() -> Self { CostFunc::MSE }
}
