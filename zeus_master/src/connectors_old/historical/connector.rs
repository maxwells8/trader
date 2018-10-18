pub mod connector {
    use super::super::*;

    impl Connector for History {
        type Args = Args;

        fn new(mut args: Self::Args) -> Self {
            let bars = args.load_series().expect("Unable to load historical data!");
            History {
                connector: args.connector,
                instrument: args.instrument,
                granularity: args.granularity,
                bars,
                current_index: Arc::new(Mutex::new(0)),
            }
        }

        fn start(&self) -> mpsc::Receiver<Bar> {
            let (channel, recv) = mpsc::sync_channel(0);
            let index = Arc::clone(&self.current_index);
            let mut bars = self.bars.clone();
            thread::spawn(move || {
                let mut start: usize = 0;
                {
                    let i = index.lock().unwrap();
                    start = (*i);
                }

                for indx in start..bars.len() {
                    channel.send(bars.remove(start));
                    //TODO There is a race condition here
                    {
                        let mut i = index.lock().unwrap();
                        *i = indx;
                    }
                }
            });
            recv
        }

        fn get_history(&self, count: usize) -> Vec<Bar> {
            if count == 0 { return vec![] }

            let mut i = self.current_index.lock().unwrap();
            let index = (*i);
            if index != 0 {
                panic!("History needs to be reset before getting history!");
            }
            else {
                let len = self.bars.len();
                let bars = if count < len { self.bars[0..count].to_owned() } else { self.bars.clone() };
                (*i) = bars.len();
                bars
            }
        }

        fn get_max_history(&self) -> Vec<Bar> {
            self.get_history(self.bars.len())
        }

        fn current_tick(&self) -> Tick {
            let i = self.current_index.lock().unwrap();
            let bar = &self.bars[(*i)];
            Tick { bid: bar.close, ask: bar.close, timestamp: bar.date + (self.granularity.seconds() as i64) }
        }

        fn take_action(&self, action: &Action) -> ActionResult {
            match action {
                &Action::Buy | &Action::Sell => {
                    let i = self.current_index.lock().unwrap();
                    let cp = (&self.bars[(*i)]).close;
                    ActionResult::Price(cp)
                },
                _ => ActionResult::None
            }
        }

        fn get_open_trades(&self) -> (Vec<f64>, Action) {
            (vec![], Action::None)
        }

        fn granularity<'g, 'c: 'g>(&'c self) -> &'g Granularity {
            &self.granularity
        }

        fn reset(&self) {
            let mut ci = self.current_index.lock().unwrap();
            (*ci) = 0;
        }

        fn save_history(&self, count: usize) {
            panic!("Cannot save history from the history connector!");
        }

        fn save_max_history(&self) {
            panic!("Cannot save history from the history connector!");
        }

        fn save_bars(&self, bars: Vec<Bar>) {
            panic!("Cannot save bars from the history connector!");
        }
    }
}