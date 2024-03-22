use crate::prelude::*;
use ndarray::{Array1, Array2, Axis};

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

pub type Gradients = Vec<Array2<f64>>;
pub type Errors = Array1<f64>;

impl MLP {
    pub fn new(layers: Vec<Layer>) -> Self {
        MLP { layers }
    }

    pub fn forward(&self, inputs: &Array1<f64>) -> Array2<f64> {
        let mut activation = inputs.clone().insert_axis(Axis(0));

        for layer in &self.layers {
            activation = layer.forward(&activation);
        }

        activation
    }

    pub fn backward(
        &self,
        inputs: &Array1<f64>,
        outputs: &Array1<f64>,
        errors: &Array1<f64>,
    ) -> (Gradients, Errors) {
        let mut errors = errors.to_owned();
        let mut gradients = Vec::new();
        for layer in self.layers.iter().rev() {
            let (errors_, gradients_, _) = layer.backward(&inputs, &outputs, &errors);
            errors = errors_;
            gradients.push(gradients_);
        }
        (gradients, errors)
    }
}
