# pyrus-nn

[![Build Status](https://milesgranger.visualstudio.com/builds/_apis/build/status/pyrus-nn?branchName=master)](https://milesgranger.visualstudio.com/builds/_build/latest?definitionId=1&branchName=master)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=milesgranger/black-jack)](https://dependabot.com)
[![crates.io](http://meritbadge.herokuapp.com/pyrus-nn)](https://crates.io/crates/pyrus-nn)

[Rust API Documentation](https://docs.rs/pyrus-nn)

Lightweight neural network framework written in Rust, with _thin_ python bindings.

- Features:
    - Serialize networks into/from YAML & JSON!
        - Rust -> serde compatible
        - Python -> `network.to_dict()` & `Sequential.from_dict()`
    - Python install requires _zero_ dependencies
    - No external system libs to install 
    
- Draw backs:
    - Only supports generic gradient descent. 
    - Fully connected (Dense) layers only so far
    - Activation functions limited to linear, tanh, sigmoid and softmax
    - Cost functions limited to MSE, MAE, Cross Entropy and Accuracy
    
### Install:

Python:
```
pip install pyrus-nn  # Has ZERO dependencies!
```

Rust:
```toml
[dependencies]
pyrus-nn = "0.2.1"
```



### From Python
```python
from pyrus_nn.models import Sequential
from pyrus_nn.layers import Dense

model = Sequential(lr=0.001, n_epochs=10)
model.add(Dense(n_input=12, n_output=24, activation='sigmoid'))
model.add(Dense(n_input=24, n_output=1, activation='sigmoid'))

# Create some X and y, each of which must be 2d
X = [list(range(12)) for _ in range(10)]
y = [[i] for i in range(10)]  

model.fit(X, y)
out = model.predict(X)

```

---

### From Rust
```rust
use ndarray::Array2;
use pyrus_nn::{network::Sequential, layers::Dense};


// Network with 4 inputs and 1 output.
fn main() {
    let mut network = Sequential::new(0.001, 100, 32, CostFunc::CrossEntropy);
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
    
    let X: Array2<f32> = ...
    let y: Array2<f32> = ...
    
    network.fit(X.view(), y.view());
    
    let yhat: Array2<f32> = network.predict(another_x.view());
}

```