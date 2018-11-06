use std::rc::Rc;
use datasource;
use brokers;
use session::{Session, RawSession};
use models::{Bar, History};
use std::mem;
use datasource::DataSource;
use brokers::Broker;
use session::ToRaw;

type D = datasource::OANDA;
type B = brokers::Sim;

export_owned!(stream_bars(bar_count: usize, callback: *const fn(*const RawSession<D, B>, *const Bar, bool) -> *const RawSession<D, B>) move |sess| {
    backtest(sess, Box::new(move |datasource| datasource.fetch_latest_history(bar_count)), callback)
});

export_owned!(stream_range(from: i64, to: i64, callback: *const fn(*const RawSession<D, B>, *const Bar, bool) -> *const RawSession<D, B>) move |sess| {
    backtest(sess, Box::new(move |datasource| datasource.fetch_history(from, to)), callback)
});

fn backtest(raw: &Rc<RawSession<D, B>>, loader: Box<Fn(&D) -> Vec<Bar>>, callback: *const fn(*const RawSession<D, B>, *const Bar, bool) -> *const RawSession<D, B>) {
    let mut sess = raw.clone();
    let callback = unsafe { *callback };
    let bars = loader(&sess.datasource);

    let mut i = 0;
    //println!("Bars: {}", bars.len());
    let last = (bars.len() as i32) - 1;
    for mut bar in bars {
        sess.broker.borrow_mut().on_bar(&bar);
        let done = i == last;
        //println!("Sending data; Bar: {:?}, Done: {}", &bar, done);
        let ptr = &mut bar as *mut Bar;
        mem::forget(bar);
        sess = Session::cooked(callback(sess.raw(), ptr, done));
        i += 1;
    }
    //println!("Stream finished!");
}

//TODO Externalize the ability to specify a time range

export!(load_history(count: usize, history: *mut History) move |sess| {
    let bars = sess.datasource.load_history(count);
    unsafe {
        let history = &mut *history;
        history.bars = bars.as_ptr();
        history.len = bars.len() as i32;
        mem::forget(history);
    };
    mem::forget(bars);
});
