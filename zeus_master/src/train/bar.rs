use super::*;
use super::super::Action;
use data::models::{Bar, Trade, Tick};
use data::utils::*;
use features::{Features, Bars};
use std::process::{Command, ExitStatus};
use json;
use std::thread::JoinHandle;
use std::thread;
use std::path::Path;
use num_cpus::get as get_num_cpu;
use std::time::SystemTime;
use ndarray as nd;
use ndarray::arr3;


macro_rules! execute {
    ( $( $x:expr ), * ) => {
        {
            let mut temp = Vec::new();
            $(
                temp.push($x);
            )*
            let cmd = temp.remove(0);
            assert!(Command::new(cmd)
                            .args(&temp)
                            .spawn()
                            .unwrap()
                            .wait()
                            .unwrap()
                            .success(),
            "External command failed to exit successfully.");
        }
    };
}

pub struct BarTrainer<F: Features> {
    file: String,
    bars: Bars<F>
}

impl<F: Features> BarTrainer<F> {
    pub fn new(file: String, bars: Bars<F>) -> Self {
        BarTrainer { file, bars }
    }

    fn close(&self, trade: &mut Trade, tick: &Tick) -> bool {
        if !trade.closed() {
            if self.bars.features().should_close(trade, tick) {
                trade.close = Some(tick.close_price(&trade.action));
            }
            else {
                return false;
            }
        }

        true
    }
}

impl<'c, C: 'c + Connector, F: Features> Trainer<'c, C> for BarTrainer<F> {
    fn train(&mut self, connector: &'c C) {
        let name = &self.bars.name();
        let script = format!("models/{}", &self.file);
        let data_file = format!("res/models/data/{}.json", name);
        let target_file = format!("res/models/target/{}.json", name);
        let model_folder = String::from("res/models/trained/");

        //TODO This is only checking the case if data and target exists but not if only one exists.
        if Path::new(&data_file).exists() && Path::new(&target_file).exists() {
            println!("Data and Target files already exists.")
        }
        else {
            println!("Removing files if they already exist.");
            fs::remove_file(&data_file);
            fs::remove_file(&target_file);

            println!("Getting history.");
            let mut bars = connector.get_max_history();

            let mut samples: Vec<Vec<Vec<f64>>> = vec![];
            let mut labels: Vec<Vec<i32>> = vec![];

            let mut actions = 0;
            println!("Loading samples and labels.");
            for _ in 0..bars.len() {
                let bar = bars.remove(0);
                let price = bar.close;
                let timestamp = bar.date;
                if let Some(sample) = self.bars.compile_input(bar) {
                    let mut buy = Trade { open: price, close: None, timestamp, action: Action::Buy };
                    let mut sell = Trade { open: price, close: None, timestamp, action: Action::Sell };

                    for index in 0..bars.len() {
                        let future = &bars[index];
                        let tick = Tick { bid: future.close, ask: future.close, timestamp: future.date + (connector.granularity().seconds() as i64) };

                        if self.close(&mut buy, &tick) && self.close(&mut sell, &tick) {
                            break;
                        }
                    }

                    //TODO Should we handle the case where only one of the trades were able to be
                    //TODO closed before the end of the series?

                    if buy.closed() && sell.closed() {
                        let action = self.bars.features().choose(&buy, &sell);
                        let is_action = if action != Action::None {
                            actions += 1;
                            true
                        } else { false };
                        if is_action || actions >= labels.len() {
                            samples.push(sample);
                            labels.push(action.as_label());
                        }
                    }
                }
            }

            println!("Calculating thread count.");
            let num_threads = get_num_cpu() - 1;
            let batch = samples.len() / num_threads;
            let mut threads: Vec<JoinHandle<_>> = vec![];
            println!("Splitting workload.");
            for i in 0..num_threads {
                let s: Vec<Vec<Vec<f64>>> = samples.drain(0..batch).collect();
                let l: Vec<Vec<i32>> = labels.drain(0..batch).collect();
                let t = thread::spawn(move || {
                    let s_bytes = json::to_vec(&s).expect("Failed to serialize samples!");
                    let l_bytes = json::to_vec(&l).expect("Failed to serialize labels!");
                    (s_bytes, l_bytes)
                });
                threads.push(t);
            }

            println!("Joining serialized samples.");
            threads.drain(0..num_threads).enumerate().for_each(|(t, trd)| {
                let (mut s_bytes, mut l_bytes) = trd.join().unwrap_or_else(|arg| {
                    panic!("Failed to join serialization thread!");
                });

                {
                    let mut df = fs::OpenOptions::new().append(true).create(true).open(&data_file).unwrap();
                    let mut tf = fs::OpenOptions::new().append(true).create(true).open(&target_file).unwrap();

                    unsafe {
                        if t == 0 {
                            let mut ch = String::from("[");
                            let lead = ch.as_bytes_mut();
                            df.write_all(lead).unwrap();
                            tf.write_all(lead).unwrap();
                        }

                        df.write_all(&mut s_bytes).unwrap();
                        tf.write_all(&mut l_bytes).unwrap();

                        if t < num_threads - 1 {
                            let mut ch = String::from(",");
                            let sep = ch.as_bytes_mut();
                            df.write_all(sep).unwrap();
                            tf.write_all(sep).unwrap();
                        }
                        else {
                            let mut ch = String::from("]");
                            let end = ch.as_bytes_mut();
                            df.write_all(end).unwrap();
                            tf.write_all(end).unwrap();
                        }
                    }
                }
            });
        }

        println!("Executing {}.", &script);
        execute!(&script, &name, &data_file, &target_file, &model_folder);
    }
}

#[derive(Serialize)]
struct Data {
    X: Vec<u8>,
    Y: Vec<u8>
}
