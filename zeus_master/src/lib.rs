#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate hyper;
#[macro_use]
extern crate serde_json as json;
extern crate serde;
extern crate reqwest;
extern crate ndarray as nd;
extern crate chrono;
extern crate uuid;
extern crate rand;

use std::iter::FromIterator;
use std::slice;
use std::sync::{Arc, Mutex, MutexGuard};

macro_rules! export(
    ($name: ident ( $($x: ident: $t: ty),* ) $body: expr) => {
        #[no_mangle]
        pub extern "C" fn $name(raw: *const RawSession<datasource::OANDA, brokers::Sim>, $($x:$t,)*) -> *const RawSession<datasource::OANDA, brokers::Sim> {
            Session::run(raw, $body)
        }
    }
);

pub mod models;
pub mod storage;
pub mod connectors;
pub mod datasource;
pub mod brokers;
pub mod session;
