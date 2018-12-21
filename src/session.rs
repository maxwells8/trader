use std::rc::Rc;
use std::cell::RefCell;

use env_logger;

use datasource;
use models::{Granularity, RawString};
use brokers;

pub type RawSession<D: datasource::DataSource, B: brokers::Broker> = Session<D, B>;

pub trait ToRaw<D: datasource::DataSource, B: brokers::Broker> {
    fn raw(self) -> *const ();
}

pub struct Session<D: datasource::DataSource, B: brokers::Broker> {
    pub marker: String,
    pub datasource: D,
    pub broker: Rc<RefCell<B>>
}

impl<D: datasource::DataSource, B: brokers::Broker> Session<D, B> {
    pub fn new(datasource: D, broker: B) -> Rc<RawSession<D, B>> {
        Rc::new(Session {
            marker: format!("{}_{}", D::name(), B::name()),
            broker: Rc::new(RefCell::new(broker)),
            datasource
        })
    }

    pub fn cooked(raw: *const ()) -> Option<Rc<Self>> {
        unsafe {
            let temp = Rc::from_raw(raw as *const Self);
            //println!("cooking: {}_{}", D::name(), B::name());
            if &temp.marker == &format!("{}_{}", D::name(), B::name()) {
                Some(temp)
            } else {
                std::mem::forget(temp);
                None
            }
        }
    }
}

impl<D: datasource::DataSource, B: brokers::Broker> ToRaw<D, B> for Rc<RawSession<D, B>> {
    fn raw(self) -> *const () {
        Rc::into_raw(self) as *const ()
    }
}

#[no_mangle]
pub extern "C" fn create_session(symbol: *const RawString, period: char, frequency: i32) -> *const () {
    env_logger::init();
    info!("Creating sim OANDA session: ");
    let instrument = unsafe { (*symbol).to_string() };
    info!("{} ", instrument);
    let granularity = Granularity::from_parts(period, frequency).unwrap();
    info!("{:?}", granularity);
    let broker = brokers::Sim::new(&instrument);
    let datasource = datasource::OANDA::new(instrument, granularity);
    Session::new(datasource, broker).raw()
}

#[no_mangle]
pub extern "C" fn create_live_session(symbol: *const RawString, period: char, frequency: i32) -> *const () {
    env_logger::init();
    info!("Creating **LIVE** OANDA session: ");
    let instrument = unsafe { (*symbol).to_string() };
    info!("{}", instrument);
    let granularity = Granularity::from_parts(period, frequency).unwrap();
    info!("{:?}", granularity);
    let broker = brokers::oanda::OANDA::new(instrument.clone());
    let datasource = datasource::OANDA::new(instrument, granularity);
    Session::new(datasource, broker).raw()
}
