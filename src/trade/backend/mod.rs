use super::*;

pub mod tensorflow;

pub trait Backend<T> {
    fn evaluate(&mut self, input: &T) -> Action;
}