use datasource;
use brokers;
use datasource::StreamType;
use session;
use session::{Session, RawSession};
use models::{Bar, History};
use std::mem;
use datasource::DataSource;
use brokers::Broker;
use session::ToRaw;

export!(stream_bars(bar_count: usize, callback: *const fn(*const Bar, bool)) move |sess| {
    backtest(sess, Box::new(move |datasource| datasource.fetch_latest_history(bar_count)), callback)
});

export!(stream_range(from: i64, to: i64, callback: *const fn(*const Bar, bool)) move |sess| {
    backtest(sess, Box::new(move |datasource| datasource.fetch_history(from, to)), callback)
});

fn backtest<D: datasource::DataSource + Send + Sync + 'static, B: brokers::Broker + Send + Sync + 'static>(sess: &Session<D, B>, loader: Box<Fn(&D) -> Vec<Bar>>, callback: *const fn(*const Bar, bool)) {
    let broker = sess.broker.clone();
    let callback = unsafe { *callback };
    sess.datasource.stream(StreamType::Sim(loader(&sess.datasource)), move |bar, done| {
        println!("On bar (Rust)");
        if let Ok(mut broker) = broker.lock() {
            broker.on_bar(&bar);
        } else { return; }

        println!("Sending data; Bar: {:?}, Done: {}", &bar, done);
        let ptr = &bar as *const Bar;
        mem::forget(bar);
        callback(ptr, done);
        println!("Callback finished");
    });
}

//TODO Externalize the ability to specify a time range

export!(load_history(count: usize, history: *mut History) move |sess| {
    let bars = sess.datasource.load_history(count);
    unsafe {
        let mut history = &mut *history;
        history.bars = bars.as_ptr();
        history.len = bars.len() as i32;
        mem::forget(history);
    };
    mem::forget(bars);
});
