use ndarray::{Array1, Array2};

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn d_relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps = x.map(|v| (v - max).exp());
    let sum: f64 = exps.iter().sum();
    exps / sum
}

pub fn d_softmax(x: &Array1<f64>) -> Array2<f64> {
    let softmax = softmax(x);
    let mut jacobian = Array2::zeros((x.len(), x.len()));
    for i in 0..x.len() {
        for j in 0..x.len() {
            jacobian[[i, j]] = if i == j {
                softmax[i] * (1.0 - softmax[i])
            } else {
                -softmax[i] * softmax[j]
            };
        }
    }
    jacobian
}
