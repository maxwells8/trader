use std::rc::Rc;
use datasource;
use brokers;
use session::{Session, RawSession};
use models::{Bar, History};
use std::mem;
use datasource::DataSource;
use brokers::Broker;
use session::ToRaw;
use std::sync::mpsc::{RecvTimeoutError};
use std::time::Duration;
use std::io;
use std::io::prelude::*;

#[export]
fn load_history(from: i64, to: i64, history: *mut History) {
    let bars = sess.datasource.load_history(from, to);
    unsafe {
        let history = &mut *history;
        history.bars = bars.as_ptr();
        history.len = bars.len() as i32;
        mem::forget(history);
    };
    mem::forget(bars);
}

#[export(sim)]
fn stream_bars(file: bool, bar_count: usize, callback: *const fn(*const (), *const Bar) -> *const ()) {
    backtest(sess,
        Box::new(move |datasource| if file { datasource.load_latest_history(bar_count) }
                                   else { datasource.fetch_latest_history(bar_count) }),
        callback
    );
}

#[export(sim)]
fn stream_range(file: bool, from: i64, to: i64, callback: *const fn(*const (), *const Bar) -> *const ()) {
    backtest(sess,
        Box::new(move |datasource| if file { datasource.load_history(from, to) }
                                   else { datasource.fetch_history(from, to) }),
        callback
    );
}

fn backtest<D: datasource::DataSource, B: brokers::Broker>
(raw: &Rc<RawSession<D, B>>, loader: Box<Fn(&D) -> Vec<Bar>>, callback: *const fn(*const (), *const Bar) -> *const ()) {
    let mut sess = raw.clone();
    let callback = unsafe { *callback };
    let bars = loader(&sess.datasource);

    //println!("Bars: {}", bars.len());
    for mut bar in bars {
        sess.broker.borrow_mut().on_bar(&bar);
        //println!("Sending data; Bar: {:?}", &bar);
        let ptr = &mut bar as *mut Bar;
        mem::forget(bar);
        sess = Session::cooked(callback(sess.raw(), ptr)).unwrap();
    }
    //println!("Stream finished!");
}

#[export]
fn stream_live(callback: *const fn(*const (), *const Bar) -> *const ()) {
    let mut sess = sess.clone();
    let callback = unsafe { *callback };

    let recv = sess.datasource.poll_latest();
    loop {
        let ptr = match recv.recv_timeout(Duration::from_secs(2)) {
            Ok(mut bar) => {
                sess.broker.borrow_mut().on_bar(&bar);

                let ptr = &mut bar as *mut Bar;
                mem::forget(bar);
                ptr
            },
            Err(RecvTimeoutError::Disconnected) => {
                break;
            },
            Err(RecvTimeoutError::Timeout) => {
                std::ptr::null()
            }
        };
        sess = Session::cooked(callback(sess.raw(), ptr)).unwrap();
    }
}
