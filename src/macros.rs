/// Create a layer of size
#[macro_export]
macro_rules! layer {
    // layer!(784, ~relu)
    ($inputs:expr, $ouputs:expr) => {
        Layer::new($inputs, $ouputs)
    };
}
