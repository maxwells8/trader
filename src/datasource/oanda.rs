use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::thread;

use storage;
use super::DataSource;
use connectors;
use datasource::StreamType;
use models::{Bar, Granularity};

pub struct OANDA {
    client: connectors::OANDA,
    instrument: String,
    granularity: Granularity
}

impl OANDA {
    pub fn new(instrument: String, granularity: Granularity) -> OANDA {
        OANDA {
            client: connectors::OANDA::client(),
            instrument,
            granularity
        }
    }
}

impl DataSource for OANDA {
    fn name() -> String {
        String::from("oanda")
    }

    fn description(&self) -> String {
        format!("{}({:?})", &self.instrument, &self.granularity)
    }

    fn granularity<'g, 's: 'g>(&'s self) -> &'g Granularity {
        &self.granularity
    }

    fn fetch_latest_history(&self, count: usize) -> Vec<Bar> {
        self.client.load_latest_history(&self.instrument, &self.granularity, count)
    }

    fn fetch_history(&self, from: i64, to: i64) -> Vec<Bar> {
        self.client.load_history_between(&self.instrument, &self.granularity, from, to)
    }

    fn poll_latest(&self) -> std::sync::mpsc::Receiver<Bar> {
        self.client.poll_latest(&self.instrument, &self.granularity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_history() {
        let oanda = OANDA {
            client: connectors::OANDA::client(),
            instrument: String::from("EUR_USD"),
            granularity: Granularity::M1
        };
        println!("{}", oanda.load_history(1104537605, 1104737600).len());
        // assert_eq!(50, oanda.load_latest_history(50).len());
        // let storage = oanda.load_from_storage();
        // assert!(storage[0].date < storage.last().unwrap().date);
    }
}
