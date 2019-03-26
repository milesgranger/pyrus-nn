use pyo3::prelude::*;

use crate::network::Sequential;
use crate::costs::CostFunc;
use crate::layers::{Layer, Dense, Activation};


#[pyclass]
pub struct PyrusDense {
    layer: Dense
}

#[pymethods]
impl PyrusDense {

    #[new]
    fn __new__(obj: &PyRawObject, n_input: usize, n_output: usize) -> PyResult<()> {
        obj.init(|_| {
            let activation = Activation::Tanh;
            PyrusDense { layer: Dense::new(n_input, n_output, activation) }
        })
    }
}

#[pyclass]
pub struct PyrusSequential {
    network: Sequential
}

#[pymethods]
impl PyrusSequential {

    #[new]
    fn __new__(obj: &PyRawObject, lr: f32, n_epoch: usize) -> PyResult<()> {
        obj.init(|_| {

            // TODO: Builder pattern
            let mut network = Sequential::new();
            network.n_epoch = n_epoch;
            network.lr = lr;

            PyrusSequential{network}
        })
    }

    fn add_dense(&mut self, n_input: usize, n_output: usize) -> PyResult<()> {
        self.network.add(Dense::new(n_input, n_output, Activation::Sigmoid)).unwrap();
        Ok(())
    }
}


#[pymodinit]
fn pyrus_nn(py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<PyrusSequential>()?;
    m.add_class::<PyrusDense>()?;
    Ok(())

}
