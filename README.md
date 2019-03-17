# pyrus-nn

[![Build Status](https://milesgranger.visualstudio.com/builds/_apis/build/status/pyrus-nn?branchName=master)](https://milesgranger.visualstudio.com/builds/_build/latest?definitionId=1&branchName=master)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=milesgranger/black-jack)](https://dependabot.com)

Lightweight neural network framework written in Rust.


```rust
use ndarray::Array2;

// Network with 4 inputs and 1 output.
let mut network = Sequential::new();
assert!(
    network.add(Dense::new(4, 5)).is_ok()
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


// Override defaults...
network.n_epoch = 10;
network.lr = 0.0001;

let X: Array2<f32> = ...
let y: Array2<f32> = ...

network.fit(X, y);

let yhat: Array2<f32> = network.predict(another_x);

```