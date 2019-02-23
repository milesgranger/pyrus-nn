use ndarray::arr2;
use float_cmp::ApproxEq;
use pretty_assertions::{assert_eq, assert_ne};

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

#[test]
fn test_mae() {
    let y_true = arr2(&[
        [1.], [2.], [3.]
    ]);
    let y_hat = arr2(&[
        [1.], [0.5], [3.5]
    ]);

    let error = costs::mean_absolute_error(y_true.view(), y_hat.view());
    assert_eq!(error, 0.6666667);
}

#[test]
fn test_accuracy_score() {
    let y_true = arr2(&[
        [0.], [1.], [1.]
    ]);
    let y_hat = arr2(&[
        [1.], [1.], [1.]
    ]);

    let error = costs::accuracy_score(y_true.view(), y_hat.view());
    assert_eq!(error, 0.6666667);

    let y_true = arr2(&[
        [0., 1.], [1., 0.], [1., 0.]
    ]);
    let y_hat = arr2(&[
        [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]
    ]);

    let error = costs::accuracy_score(y_true.view(), y_hat.view());
    assert_eq!(error, 0.6666667);
}

#[test]
fn test_cross_entropy() {
    let y_true = arr2(&[
        [0., 1.], [1., 0.], [1., 0.]
    ]);
    let y_hat = arr2(&[
        [1., 0.], [1., 0.], [1., 0.]
    ]);

    let error = costs::cross_entropy(y_true.view(), y_hat.view());
    assert_eq!(error, 11.512925);
}