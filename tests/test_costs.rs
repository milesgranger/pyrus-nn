use ndarray::arr2;

use pyrus_nn::costs;

#[test]
fn test_mse() {
    let y_true = arr2(&[
        [1.], [2.], [3.]
    ]);
    let y_hat = arr2(&[
        [1.], [0.5], [3.5]
    ]);

    let error = costs::mean_squared_error(y_true.view(), y_hat.view());
    assert_eq!(error, 0.833333333);
}