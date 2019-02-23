use ndarray::arr2;
use float_cmp::ApproxEq;
use pretty_assertions::{assert_eq, assert_ne};

use pyrus_nn::activations;


#[test]
fn test_tanh() {
    let x = arr2(&[
        [3.5]
    ]);
    let result = activations::tanh(&x, false);
    assert_eq!(result.sum(), 0.99817795);
}

#[test]
fn test_sigmoid() {
    let x = arr2(&[
        [3.5]
    ]);
    let result = activations::sigmoid(&x, false);
    assert_eq!(result.sum(), 0.97068775);
}

#[test]
fn test_softmax() {
    let x = arr2(&[
        [1., 2., 3.]
    ]);
    let result = activations::softmax(&x, false);
    assert_eq!(result, arr2(&[[0.09003057, 0.24472848, 0.66524094]]));
}