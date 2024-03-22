use crate::layers::LayerError;
use ndarray::{Array1, Array2};

pub(crate) fn random_array1(size: usize) -> Array1<f64> {
    Array1::from_shape_vec(size, (0..size).map(|_| rand::random::<f64>()).collect()).unwrap()
}

pub(crate) fn random_array2(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_vec(
        (rows, cols),
        (0..rows * cols).map(|_| rand::random::<f64>()).collect(),
    )
    .unwrap()
}

pub(crate) fn xavier_array2(rows: usize, cols: usize) -> Array2<f64> {
    let std_dev = (2.0 / (rows as f64 + cols as f64)).sqrt();
    Array2::from_shape_vec(
        (rows, cols),
        (0..rows * cols)
            .map(|_| rand::random::<f64>() * 2.0 * std_dev - std_dev)
            .collect(),
    )
        .unwrap()
}

pub fn dims_tuple(matrix: &Array2<f64>) -> (usize, usize) {
    (matrix.shape()[0], matrix.shape()[1])
}

/// Checks if the inputs and weights are compatible for matrix multiplication without transposing
pub(crate) fn mmult_compat(inputs: &Array2<f64>, weights: &Array2<f64>) -> Result<(), LayerError> {
    match inputs.shape()[1] == weights.shape()[0] || inputs.shape()[0] == weights.shape()[0] {
        true => return Ok(()),
        false => {
            log::error!("Inputs and weights are not compatible");
            log::warn!("Inputs: {:?}", inputs.shape());
            log::warn!("Weights: {:?}", weights.shape());
            return Err(LayerError::IncompatibleInput {
                inputs: super::util::dims_tuple(inputs),
                weights: super::util::dims_tuple(weights),
            });
        }
    }
}
