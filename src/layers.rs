use ndarray::{Array1, Array2};

/// Layer operations
pub trait LayerExt {
    /// Forward pass
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64>;
    /// Backward pass
    fn backward(
        &self,
        inputs: &Array1<f64>,
        outputs: &Array1<f64>,
        errors: &Array1<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<f64>);
}

#[derive(Debug)]
pub struct Layer {
    /// Number of inputs to the layer
    in_features: usize,
    /// Number of neurons in the layer
    out_features: usize,
    /// Weights of the layer
    weights: Array2<f64>,
    /// Biases of the layer
    biases: Array1<f64>,
    /// Activation function of the layer
    activation: fn(f64) -> f64,
    /// Derivative of the activation function
    d_activation: Option<fn(f64) -> f64>,
}

impl Layer {
    /// Activation defaults to ReLU
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weights = super::util::xavier_array2(in_features, out_features);
        let biases = super::util::random_array1(out_features);
        let activation = super::activation::relu;
        let layer = Layer {
            in_features,
            out_features,
            weights,
            biases,
            activation,
            d_activation: Some(super::activation::d_relu),
        };
        log::debug!("Layer created: {:?}", layer);
        layer
    }

    /// Set the activation function
    pub fn with_activation(mut self, activation: fn(f64) -> f64) -> Self {
        self.activation = activation;
        self
    }
}

#[derive(Debug)]
pub enum LayerError {
    IncompatibleInput {
        inputs: (usize, usize),
        weights: (usize, usize),
    },
}

impl LayerExt for Layer {
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        match super::util::mmult_compat(inputs, &self.weights) {
            Ok(_) => {
                let z = inputs.dot(&self.weights) + &self.biases;
                return z.mapv(self.activation);
            }
            Err(_) => panic!("Inputs and weights are not compatible"),
        }
    }

    fn backward(
        &self,
        inputs: &Array1<f64>,
        outputs: &Array1<f64>,
        errors: &Array1<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let d_activation = self.d_activation.unwrap();
        let gradients = errors * &outputs.mapv(d_activation);

        // Manually compute the outer product for d_weights
        let d_weights = Array2::from_shape_fn((self.out_features, self.in_features), |(i, j)| {
            gradients[i] * inputs[j]
        });

        let d_biases = gradients.clone();
        let d_inputs = self.weights.t().dot(&gradients);
        (d_inputs, d_weights, d_biases)
    }
}
