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

        #[rustfmt::skip]
        let layers = vec![
            layer!(784, 128), 
            layer!(128, 64), 
            layer!(64, 10)
        ];

        let mlp = MLP::new(layers);
        let inputs = util::random_array1(784);
        let outputs = mlp.forward(&inputs);

        log::info!("Outputs {:?}", outputs);

    }
}
