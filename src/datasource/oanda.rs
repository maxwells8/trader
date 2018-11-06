use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::thread;

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
    fn name(&self) -> String {
        String::from("oanda")
    }

    fn description(&self) -> String {
        format!("{}|{:?}", &self.instrument, &self.granularity)
    }

    fn fetch_latest_history(&self, count: usize) -> Vec<Bar> {
        let bars = self.client.load_latest_history(&self.instrument, &self.granularity, count);
        bars
    }

    fn fetch_history(&self, from: i64, to: i64) -> Vec<Bar> {
        self.client.load_history_between(&self.instrument, &self.granularity, from, to)
    }
}
