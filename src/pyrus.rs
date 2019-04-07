use ndarray::{Array, Array2};
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde_derive::{Deserialize, Serialize};
use serde_json;
use serde_yaml;

use crate::costs::CostFunc;
use crate::layers::{Activation, Dense, Layer};
use crate::network::Sequential;

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct PyrusSequential {
    network: Sequential,
}

#[pymethods]
impl PyrusSequential {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        lr: f32,
        n_epoch: usize,
        batch_size: usize,
        cost_func: String,
    ) -> PyResult<()> {
        obj.init(|_| {
            let cost_func = CostFunc::from(cost_func);
            let network = Sequential::new(lr, n_epoch, batch_size, cost_func);
            PyrusSequential { network }
        })
    }

    fn to_yaml(&self) -> PyResult<String> {
        Ok(serde_yaml::to_string(&self).unwrap())
    }

    #[classmethod]
    fn from_yaml(_cls: &PyType, conf: String) -> PyResult<PyrusSequential> {
        Ok(serde_yaml::from_str(&conf).unwrap())
    }

    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).unwrap())
    }

    #[classmethod]
    fn from_json(_cls: &PyType, conf: String) -> PyResult<PyrusSequential> {
        Ok(serde_json::from_str(&conf).unwrap())
    }

    fn add_dense(&mut self, n_input: usize, n_output: usize, activation: String) -> PyResult<()> {
        self.network
            .add(Dense::new(n_input, n_output, Activation::from(activation)))
            .unwrap();
        Ok(())
    }

    fn fit(&mut self, x: Vec<Vec<f32>>, y: Vec<Vec<f32>>) -> PyResult<()> {
        let x: Array2<f32> = vec2d_into_array2d(x);
        let y: Array2<f32> = vec2d_into_array2d(y);
        self.network.fit(x.view(), y.view());
        Ok(())
    }

    fn predict(&mut self, x: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let x: Array2<f32> = vec2d_into_array2d(x);
        let out = self
            .network
            .forward(x.view())
            .outer_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<Vec<f32>>>();
        Ok(out)
    }
}

// Helper, create an ndarry::Array2 from 2d Vector
fn vec2d_into_array2d(vec: Vec<Vec<f32>>) -> Array2<f32> {
    let shape = (vec.len(), vec[0].len());
    Array::from_iter(vec.into_iter().flat_map(|v| v))
        .into_shape(shape)
        .unwrap()
}

#[pymodinit]
fn pyrus_nn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyrusSequential>()?;
    Ok(())
}
