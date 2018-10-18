use std::iter::FromIterator;
use std::slice;
use std::sync::{Arc, Mutex};

use datasource;
use models::{Granularity, RawString};
use brokers;

pub type RawSession<D: datasource::DataSource, B: brokers::Broker> = Mutex<Session<D, B>>;

pub trait ToRaw<D: datasource::DataSource, B: brokers::Broker> {
    fn raw(self) -> *const RawSession<D, B>;
}

pub struct Session<D: datasource::DataSource, B: brokers::Broker> {
    pub datasource: D,
    pub broker: Arc<Mutex<B>>
}

impl<D: datasource::DataSource, B: brokers::Broker> Session<D, B> {
    pub fn run<F>(raw: *const Mutex<Self>, func: F) -> *const Mutex<Self>
    where F: Fn(&Self) {
        let sess = Self::cooked(raw);
        if let Ok(sess) = sess.lock() {
            func(&sess)
        }
        sess.raw()
    }

    pub fn cooked(raw: *const Mutex<Self>) -> Arc<Mutex<Self>> {
        unsafe { Arc::from_raw(raw) }
    }
}

impl Session<datasource::OANDA, brokers::Sim> {
    pub fn new(instrument: String, granularity: Granularity) -> Arc<RawSession<datasource::OANDA, brokers::Sim>> {
        Arc::new(Mutex::new(Session {
            datasource: datasource::OANDA::new(instrument, granularity),
            broker: Arc::new(Mutex::new(brokers::Sim::new()))
        }))
    }
}

impl<D: datasource::DataSource, B: brokers::Broker> ToRaw<D, B> for Arc<RawSession<D, B>> {
    fn raw(self) -> *const RawSession<D, B> {
        Arc::into_raw(self)
    }
}

#[no_mangle]
pub extern "C" fn create_session(symbol: *const RawString, period: char, frequency: i32) -> *const RawSession<datasource::OANDA, brokers::Sim> {
    print!("Creating sim OANDA session: ");
    let instrument = unsafe { (*symbol).to_string() };
    print!("{}; ", instrument);
    if !instrument.ends_with("USD") { panic!("Only USD based currency pairs are supported!") }
    let granularity = Granularity::from_parts(period, frequency).unwrap();
    println!("{:?}", granularity);
    Session::new(instrument, granularity).raw()
}
