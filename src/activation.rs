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
