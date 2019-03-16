
use pyo3::prelude::*;

use crate::network::Sequential;
use crate::costs::CostFunc;


#[pyclass]
pub struct PyrusSequential {
    network: Sequential
}

#[pymethods]
impl PyrusSequential {

    #[new]
    fn __new__(obj: &PyRawObject, lr: f32) -> PyResult<()> {
        obj.init(|_| PyrusSequential{ network: Sequential::new() })
    }

}


#[pymodinit]
fn pyrus_nn(py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<PyrusSequential>()?;
    Ok(())

}
