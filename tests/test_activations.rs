use ndarray::arr2;
use float_cmp::ApproxEq;

use pyrus_nn::activations;


#[test]
fn test_tanh() {
    let x = arr2(&[
        [3.5]
    ]);
    let result = activations::tanh(&x, false);

}