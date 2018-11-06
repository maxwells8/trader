use models::*;
use storage;

pub mod oanda;
pub mod external;
pub use self::oanda::OANDA;

pub enum StreamType {
    Sim(Vec<Bar>),
    Live
}

pub trait DataSource {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn fetch_latest_history(&self, count: usize) -> Vec<Bar>;
    fn fetch_history(&self, from: i64, to: i64) -> Vec<Bar>;

    fn load_history(&self, count: usize) -> Vec<Bar> {
        let file_name = format!("{}|{}", self.description(), count);
        match storage::load_bars(&self.name(), &file_name) {
            Some(bars) => bars,
            None => {
                let bars = self.fetch_latest_history(count);
                storage::save_bars(&self.name(), &file_name, &bars);
                bars
            }
        }
    }

    fn stream<F: Fn(Bar, bool) + 'static>(&self, stream_type: StreamType, on_tick: F) {
        match stream_type {
            StreamType::Sim(bars) => {
                let mut i = 0;
                println!("Bars: {}", bars.len());
                let last = (bars.len() as i32) - 1;
                for bar in bars {
                    on_tick(bar, i == last);
                    i += 1;
                }
                println!("stream finished!");
            },
            _ => unimplemented!()
        }
    }
}
