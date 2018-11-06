use std::rc::Rc;
use std::cell::RefCell;

use datasource;
use models::{Granularity, RawString};
use brokers;

pub type RawSession<D: datasource::DataSource, B: brokers::Broker> = Session<D, B>;

pub trait ToRaw<D: datasource::DataSource, B: brokers::Broker> {
    fn raw(self) -> *const RawSession<D, B>;
}

pub struct Session<D: datasource::DataSource, B: brokers::Broker> {
    pub datasource: D,
    pub broker: Rc<RefCell<B>>
}

impl<D: datasource::DataSource, B: brokers::Broker> Session<D, B> {
    pub fn run<F>(raw: *const Self, func: F) -> *const Self
    where F: Fn(&Self) {
        let sess = Self::cooked(raw);
        func(&sess);
        sess.raw()
    }

    pub fn cooked(raw: *const Self) -> Rc<Self> {
        unsafe { Rc::from_raw(raw) }
    }
}

impl Session<datasource::OANDA, brokers::Sim> {
    pub fn new(instrument: String, granularity: Granularity) -> Rc<RawSession<datasource::OANDA, brokers::Sim>> {
        Rc::new(Session {
            datasource: datasource::OANDA::new(instrument, granularity),
            broker: Rc::new(RefCell::new(brokers::Sim::new()))
        })
    }
}

impl<D: datasource::DataSource, B: brokers::Broker> ToRaw<D, B> for Rc<RawSession<D, B>> {
    fn raw(self) -> *const RawSession<D, B> {
        Rc::into_raw(self)
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
