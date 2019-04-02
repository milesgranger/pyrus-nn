use pyo3::prelude::*;
use ndarray::{Array2, Array};

use crate::network::Sequential;
use crate::layers::{Layer, Dense, Activation};


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

    fn fit(&mut self, x: Vec<Vec<f32>>, y: Vec<Vec<f32>>) -> PyResult<()> {
        let xshape = (x.len(), x[0].len());
        let yshape = (y.len(), y[0].len());
        let x: Array2<f32> = Array::from_iter(x.into_iter().flat_map(|v| v)).into_shape(xshape).unwrap();
        let y: Array2<f32> = Array::from_iter(y.into_iter().flat_map(|v| v)).into_shape(yshape).unwrap();
        self.network.fit(x.view(), y.view());
        Ok(())
    }
}

#[pymodinit]
fn pyrus_nn(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<PyrusSequential>()?;
    Ok(())

}
