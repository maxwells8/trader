use Action;
use data::models::{Trade, Tick, Bar, TradingHours};
use data::utils::*;
use std::fmt::{Display, Formatter, Error};

mod ohlc;
mod percent;

pub use self::ohlc::OHLC;
pub use self::percent::PercentChange;

pub struct Bars<F: Features> {
    features: F,
    vector: Vec<Bar>
}

impl<F: Features> Bars<F> {
    pub fn new(features: F) -> Self {
        Bars {
            features,
            vector: vec![]
        }
    }

    pub fn features<'f, 's: 'f>(&'s self) -> &'f F {
        &self.features
    }

    pub fn name(&self) -> String {
        format!("{}", self.features)
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn compile_input(&mut self, bar: Bar) -> Option<Vec<Vec<f64>>> {
        if bar.complete {
            self.vector.push(bar);
            self.features.compile_input(&self.vector)
        }
        else {
            None
        }
    }
}

pub trait Features: Display {
    fn compile_input(&self, bars: &Vec<Bar>) -> Option<Vec<Vec<f64>>>;
    fn should_close(&self, trade: &Trade, tick: &Tick) -> bool;
    fn choose(&self, first: &Trade, second: &Trade) -> Action;
}