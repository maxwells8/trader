use super::{Action, ActionResult, Connector};
use data::models::{Bar, Granularity, Tick};

use std::sync::{mpsc, Arc, Mutex};
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::str::{self, FromStr};
use std::thread;
use std::time;
use std::fmt::{self, Display, Debug, Formatter};

use json::{self, Value, Error};
use reqwest::{self as http, Client, header, StatusCode, Response};
use chrono;
use hyper::header::Headers;
use rand::distributions::{IndependentSample, Range};
use rand;

mod internal;
mod connector;

const REST: &str = "https://api-fxpractice.oanda.com";
const STREAM: &str = "https://stream-fxpractice.oanda.com";

header! { (AcceptDatetimeFormat, "Accept-Datetime-Format") => [String] }
header! { (ContentType, "Content-Type") => [String] }

#[derive(Debug)]
pub enum Mode {
    Live,
    Sim(Option<Range<u32>>)
}

pub struct Args {
    pub instrument: String,
    pub granularity: Granularity,
    pub mode: Mode
}

pub struct OANDA {
    instrument: String,
    granularity: Granularity,
    token: String,
    account: String,
    history: Arc<Mutex<Vec<Bar>>>,
    current_tick: Arc<Mutex<Tick>>,
    mode: Mode
}

impl Display for OANDA {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "oanda/{}|{:?}", &self.instrument, &self.granularity)
    }
}

impl Debug for OANDA {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "OANDA {{ instrument: {}, granularity: {:?}, mode: {:?} }}", &self.instrument, &self.granularity, &self.mode)
    }
}

fn load_credentials() -> (String, String) {
    let path = Path::new("res/creds/oanda.json");
    if let Ok(file) = File::open(&path) {
        let creds: Result<Value, Error> = json::from_reader(BufReader::new(file));
        if let Ok(creds) = creds {
            if let Some(token) = creds["token"].as_str() {
                if let Some(account) = creds["account"].as_str() {
                    return (token.to_owned(), account.to_owned())
                }
            }
        }
    }
    panic!("Failed to load OANDA credentials")
}

trait ParseValue {
    fn parse_as<T: FromStr>(&self) -> Option<T>;
    fn parse_as_f64(&self) -> Option<f64>;
    fn parse_as_i32(&self) -> Option<i32>;
}

impl ParseValue for Value {
    fn parse_as<T: FromStr>(&self) -> Option<T> {
        if let Some(s) = self.as_str() {
            if let Ok(f) = s.parse() {
                return Some(f);
            }
        }
        None
    }

    fn parse_as_f64(&self) -> Option<f64> {
        self.parse_as::<f64>()
    }

    fn parse_as_i32(&self) -> Option<i32> {
        self.parse_as::<i32>()
    }
}