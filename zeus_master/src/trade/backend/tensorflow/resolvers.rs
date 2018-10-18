use super::*;

pub trait Resolver {
    fn evaluate<T: tf::TensorType + Num + PartialOrd>(&self, tensor: &tf::Tensor<T>) -> Action;
}

pub struct Classifier;
impl Resolver for Classifier {
    fn evaluate<T: tf::TensorType + Num + PartialOrd>(&self, tensor: &tf::Tensor<T>) -> Action {
        Action::from(tensor.iter().arg_max())
    }
}