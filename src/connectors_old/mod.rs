pub mod oanda;
pub mod historical;

use data::models::{Bar, Tick, Granularity};
use super::{Action, ActionResult};
use std::sync::{mpsc, Arc, Mutex};
use std::fmt::{Display, Debug};
use std::path::Path;
use std::fs::File;
use std::io::Write;
use std::thread;
use bus::{Bus, BusReader};

use json;

pub trait Connector: Display + Debug + Send + Sync {
    type Args;
    fn new(args: Self::Args) -> Self;
    fn start(&self) -> mpsc::Receiver<Bar>;
    fn get_history(&self, count: usize) -> Vec<Bar>;
    fn get_max_history(&self) -> Vec<Bar>;
    fn current_tick(&self) -> Tick;
    fn take_action(&self, action: &Action) -> ActionResult;
    fn get_open_trades(&self) -> (Vec<f64>, Action);
    fn granularity<'g, 'c: 'g>(&'c self) -> &'g Granularity;
    fn reset(&self);

    fn save_history(&self, count: usize) {
        self.save_bars(self.get_history(count));
    }

    fn save_max_history(&self) {
        self.save_bars(self.get_max_history());
    }

    fn save_bars(&self, bars: Vec<Bar>) {
        let file_name = &format!("res/series/{}.json", self);
        let path = Path::new(file_name);
        let json = json::to_string(&bars).expect("Failed to serialize json");
        let mut file = File::create(path).expect("Failed to open file");
        file.write_all(json.as_bytes()).expect("Failed to write to file");
    }

    fn can_run(&self) -> bool { false }
    fn run(&self) -> Option<(Tick, Bar)> { None }
}

struct Connection<C: Connector> {
    bus: Arc<Mutex<Bus<(Tick, Bar)>>>,
    connector: Arc<C>
}

impl<A, C: 'static + Connector<Args = A>> Connection<C> {
    fn new(args: A) -> Self {
        Connection {
            bus: Arc::new(Mutex::new(Bus::new(10))),
            connector: Arc::new(C::new(args))
        }
    }

    fn subscribe(&self) -> Result<(Arc<C>, BusReader<(Tick, Bar)>), String> { //TODO Errors shouldn't be string values
        self.bus.lock()
                .map_err(|e| format!("{}", e))
                .and_then(|mut l| Ok((self.connector.clone(), l.add_rx())))
    }

    fn start(self) {
        thread::spawn(move || {
            while self.connector.can_run() {
                if let Some(msg) = self.connector.run() {
                    match self.bus.lock() {
                        Ok(mut b) => b.broadcast(msg),
                        _ => {}
                    }
                }
            }
        });
    }
}