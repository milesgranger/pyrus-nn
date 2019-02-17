use ndarray::{ArrayView2, Zip};


/// Determine the Mean Squared Error between `expected` and `output`
pub fn mean_squared_error(y_true: ArrayView2<f32>, y_hat: ArrayView2<f32>) -> f32 {
    y_true.iter()
        .zip(y_hat.iter())
        .map(|(yt, yh)| (yt - yh).powf(2.0))
        .sum::<f32>() / y_true.rows() as f32
}