use super::{Action, Connector};
use std::slice::Iter;
use std::cmp::PartialOrd;
use num::Num;
use tensorflow as tf;

pub fn panic_if_trades_open<'a, C: 'a + Connector>(connector: &'a C) {
    let (trades, action) = connector.get_open_trades();
    if trades.len() != 0 || action != Action::None {
        panic!("Trades are already open, please close the trades and try again!");
    }
}

pub trait ArgMax {
    fn arg_max(&mut self) -> usize;
}

impl<'a, T: 'a + Num + PartialOrd> ArgMax for Iter<'a, T> {
    fn arg_max(&mut self) -> usize {
        let mut max = &T::zero();
        let mut max_index: usize = 0;
        let mut index = 0;
        while let Some(item) = self.next() {
            if item > max {
                max = item;
                max_index = index;
            }
            index += 1;
        }
        max_index
    }
}

pub trait Convert<T> {
    fn convert(item: T) -> Self;
}

impl Convert<Vec<Vec<f64>>> for tf::Tensor<f32> {
    fn convert(item: Vec<Vec<f64>>) -> tf::Tensor<f32> {
        let size = item.len();
        let mut tensor = tf::Tensor::<f32>::new(&[1, size as u64, item[0].len() as u64]);
        for i in 0..size {
            let first = i * 5;
            let bar = &item[i];
            for b in 0..bar.len() {
                tensor[first + b] = bar[b] as f32;
            }
        }
        tensor
    }
}

impl Convert<Vec<Vec<f64>>> for tf::Tensor<f64> {
    fn convert(item: Vec<Vec<f64>>) -> tf::Tensor<f64> {
        let size = item.len();
        let mut tensor = tf::Tensor::<f64>::new(&[1, size as u64, item[0].len() as u64]);
        for i in 0..size {
            let first = i * 5;
            let bar = &item[i];
            for b in 0..bar.len() {
                tensor[first + b] = bar[b] as f64;
            }
        }
        tensor
    }
}