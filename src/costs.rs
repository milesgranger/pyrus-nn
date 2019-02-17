use ndarray::{ArrayView2, Zip};


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

/// Cost function selection `enum`
pub enum CostFunc {
    MSE,
    MAE,
}

impl std::default::Default for CostFunc {
    fn default() -> Self { CostFunc::MSE }
}
