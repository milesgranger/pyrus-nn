use ndarray::{arr2, arr1};

use pyrus_nn::network::Sequential;
use pyrus_nn::layers::{Layer, Dense};


#[test]
fn test_setup() {

    let mut network = Sequential::new();
    assert!(
        network.add(Dense::new(3, 5)).is_ok()
    );
    assert!(
        network.add(Dense::new(5, 6)).is_ok()
    );
    assert!(
        network.add(Dense::new(6, 4)).is_ok()
    );
    assert!(
        network.add(Dense::new(4, 1)).is_ok()
    );

    if let Ok(_) = network.add(Dense::new(5, 2)) {
        panic!("Added layer with mismatched sizes!")
    }

    let x = arr2(&[
        [1., 2., 3.],
        [4., 5., 6.],
        [1., 2., 3.],
        [4., 5., 6.],
        [1., 2., 3.],
        [4., 5., 6.],
        [1., 2., 3.],
        [4., 5., 6.],
    ]);
    let y = arr2(&[
        [1.],
        [0.],
        [1.],
        [0.],
        [1.],
        [0.],
        [1.],
        [0.],
    ]);
    let out = network.predict(x.clone());
    println!("Output before back prop: {:#?}", out);

    network.fit(x.clone(), y);
    let out = network.predict(x);
    println!("Output after back prop: {:#?}", out);
    // Array of two predictions
    //assert_eq!(out.shape(), &[1, 2]);
}
