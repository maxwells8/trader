#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate hyper;
#[macro_use]
extern crate serde_json as json;
extern crate serde_pickle as pickle;
extern crate serde;
extern crate rand;
extern crate tensorflow;
extern crate ansi_term;
extern crate reqwest;
extern crate chrono;
extern crate num;
extern crate num_cpus;
extern crate libc;
extern crate libloading;
extern crate bus;
#[macro_use]
extern crate ndarray;

#[macro_use]
mod colors;

#[allow(dead_code)]
#[macro_use]
mod data;
#[allow(dead_code)]
mod connectors;
#[allow(dead_code)]
mod features;
#[allow(dead_code)]
mod train;
#[allow(dead_code)]
mod trade;


use connectors::Connector;
use connectors::oanda::{OANDA, self};
use connectors::historical::{History, self};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use data::models::{Bar, Granularity};
use rand::distributions::Range;
use std::fs::File;
use std::io::Write;

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum Action {
    Buy,
    Sell,
    None
}

impl Action {
    fn flip(&self) -> Action {
        match self {
            &Action::Buy => Action::Sell,
            &Action::Sell => Action::Buy,
            _ => Action::None
        }
    }

    fn to_int(&self) -> i32 {
        match self {
            &Action::Buy => 0,
            &Action::Sell => 1,
            &Action::None => 2
        }
    }

    fn as_label(&self) -> Vec<i32> {
        match self {
            &Action::Buy => vec![1, 0, 0],
            &Action::Sell => vec![0, 1, 0],
            &Action::None => vec![0, 0, 1]
        }
    }
}

impl<T: num::Num> From<T> for Action {
    fn from(number: T) -> Self {
        if number == T::zero() {
            Action::Buy
        }
            else if number == T::one() {
                Action::Sell
            }
                else {
                    Action::None
                }
    }
}

pub enum ActionResult {
    Price(f64),
    None
}

fn oanda() -> OANDA {
    let args = oanda::Args {
        instrument: "USD_JPY".to_owned(),
        granularity: Granularity::M15,
        mode: oanda::Mode::Sim(Some(Range::new(0u32, 2000u32)))
    };
    OANDA::new(args)
}

fn history() -> History {
    let args = historical::Args {
        instrument: "EUR_USD".to_owned(),
        granularity: Granularity::M15,
        connector: "oanda".to_owned()
    };
    History::new(args)
}

fn save() {
    oanda().save_max_history()
}

fn run() {
    let mut connector = history();
    println!("Loaded Connector: {:?}", connector);
    let trading_hours: Option<data::models::TradingHours> = None; //Some(data::models::TradingHours { start: 12, end: 21 });
    let features = features::OHLC::new("something-test-1", 15.0, -7.0, 13500, 250, trading_hours);
    let train_args = features::Bars::new(features.clone());
    let trade_args = features::Bars::new(features.clone());

    let mut trainer = train::bar::BarTrainer::new("keras/cnn_lstm".to_owned(), train_args);//::external::ExternalTrainer::new("keras/cnn_lstm".to_owned(), train_args);
    train::Trainer::train(&mut trainer, &connector);

    /*
    connector.reset();

    let backend = trade::backend::tensorflow::Tensorflow::new(&trade_args.name(), trade::backend::tensorflow::resolvers::Classifier);
    let mut trader = trade::bar::BarTrader::<_, tensorflow::Tensor<f32>, _>::new(trade_args, backend, &connector);
    let mut trade_manager = trade::management::FIFO::new(true);
    let recv = connector.start();

    let mut metrics = String::new();
    println!(); //Extra whitespace
    for bar in recv {
        metrics = trade::Trader::evaluate(&mut trader, &connector, &mut trade_manager, bar);
    }
    println!("\n"); //Extra whitespace

    let mut file = File::create(&format!("res/models/metrics/{}.txt", features.name())).unwrap();
    file.write_all(unsafe { metrics.as_bytes_mut() }).unwrap();
    */
}

fn main() {
    //save();
    run();
}