# pyrus-nn

[![Build Status](https://milesgranger.visualstudio.com/builds/_apis/build/status/pyrus-nn?branchName=master)](https://milesgranger.visualstudio.com/builds/_build/latest?definitionId=1&branchName=master)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=milesgranger/black-jack)](https://dependabot.com)

Lightweight neural network framework written in Rust, with _thin_ python bindings.

### Install:

Python:
```
pip install pyrus-nn  # Has ZERO dependencies!
```

Rust:
```toml
[dependencies]
pyrus-nn = "0.2.0"
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