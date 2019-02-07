use ndarray::arr2;

use pyrus_nn::network::Sequential;
use pyrus_nn::layers::{Layer, Dense};


#[test]
fn test_setup() {

    let mut network = Sequential::new();
    assert!(
        network.add(Dense::new(3, 2)).is_ok()
    );
    assert!(
        network.add(Dense::new(2, 1)).is_ok()
    );

    if let Ok(_) = network.add(Dense::new(5, 2)) {
        panic!("Added layer with mismatched sizes!")
    }

    let x = arr2(&[
        [1., 2., 3.],
        [4., 5., 6.]
    ]);
    let out = network.predict(x);

    // Array of two predictions
    assert_eq!(out.shape(), &[1, 2]);
    println!("Output: {:#?}", out);
}
