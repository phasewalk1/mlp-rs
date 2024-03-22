pub mod activation;
pub mod layers;
pub mod macros;
pub mod mlp;
pub mod util;

pub mod prelude {
    pub use super::layers::*;
    pub use super::mlp::{Errors, Gradients, MLP};
}

#[cfg(test)]
mod tests {
    use super::layer;
    use super::prelude::*;
    use super::util;

    #[test]
    fn test_forward_pass() {
        pretty_env_logger::try_init().ok();

        let layers = vec![layer!(784, 128), layer!(128, 64), layer!(64, 10)];

        let inputs = vec![util::random_array1(784); 10];
        let mlp = MLP::new(layers);
        let mut outputs: Vec<_> = Vec::new();

        for batch in inputs.iter() {
            outputs.push(mlp.forward(batch));
        }

        log::info!("Outputs {:?}", outputs);

    }
}
