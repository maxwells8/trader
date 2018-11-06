use {Action, ActionResult};
use connectors::Connector;
use data::models::{Bar, Granularity, Tick};

use std::sync::{Arc, Mutex};
use std::fmt::{self, Formatter, Display, Debug};
use std::fs::File;
use std::io::{Write, Result, BufReader};
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use json;

mod connector;

pub struct Args {
    pub instrument: String,
    pub granularity: Granularity,
    pub connector: String
}

impl Args {
    pub fn load_series(&mut self) -> Result<Vec<Bar>> {
        let file_name = format!("res/series/{}/{}|{:?}.json", &self.connector, &self.instrument, &self.granularity);
        let file = File::open(Path::new(&file_name))?;
        Ok(json::from_reader(BufReader::new(file))?)
    }
}

pub struct History {
    connector: String,
    instrument: String,
    granularity: Granularity,
    bars: Vec<Bar>,
    current_index: Arc<Mutex<usize>>
}


impl Display for History {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Debug for History {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "History {{ connector: {}, instrument: {}, granularity: {:?} }}", &self.connector, &self.instrument, &self.granularity)
    }
}
