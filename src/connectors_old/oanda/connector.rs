pub mod connector {
    use super::super::*;

    impl Connector for OANDA {
        type Args = Args;

        fn new(args: Self::Args) -> Self {
            let (token, account) = load_credentials();
            OANDA {
                instrument: args.instrument,
                granularity: args.granularity,
                token,
                account,
                history: Arc::new(Mutex::new(vec![])),
                current_tick: Arc::new(Mutex::new(Tick { bid: 0.0, ask: 0.0, timestamp: 0 })),
                mode: args.mode
            }
        }

        fn start(&self) -> mpsc::Receiver<Bar> {
            let (channel, recv) = mpsc::channel();
            let instrument = self.instrument.clone();
            let granularity = self.granularity.clone();
            let history = Arc::clone(&self.history);
            let token = self.token.clone();
            thread::spawn(move || {
                loop {
                    //Get the most recent two bars
                    let bars = OANDA::get_history(&token, &instrument, &granularity, 2);

                    //Check if the completed bars are in history list
                    {
                        match history.lock() {
                            Ok(mut history) => {
                                let history = &mut (*history);
                                let last_timestamp = match history.last() {
                                    Some(bar) => bar.date,
                                    None => 0
                                };
                                //If not add them and send them through the channel
                                for bar in &bars {
                                    if bar.date > last_timestamp && bar.complete {
                                        history.push(bar.clone()); //NOTE: get_history should be called first?
                                        //println!("Sending {:?}", bar);
                                        &channel.send(bar.clone()); //TODO Handle possible error
                                    }
                                }
                            }
                            Err(e) => continue
                        }
                    }

                    //Check time of next incomplete bar
                    if let &Some(current) = &bars.last() {
                        if !current.complete {
                            let now = chrono::Utc::now().timestamp();
                            let sleep_time = current.date + (granularity.seconds() as i64) - now - 3;
                            if sleep_time > 0 {
                                //println!("Sleeping for {} seconds", sleep_time);
                                thread::sleep(time::Duration::from_secs(sleep_time as u64));
                            }
                        }
                    }
                }
            });

            //Receive quotes
            self.receive_ticks();

            recv
        }

        fn get_history(&self, count: usize) -> Vec<Bar> {
            if let Ok(mut lock) = self.history.lock() {
                let mut history = &mut (*lock);
                if count > history.len() {
                    history.clear();
                    //Add one to count to accommodate the incomplete bar
                    let more_history = OANDA::get_history(&self.token, &self.instrument, &self.granularity, if count >= 5000 { 5000 } else { count + 1 });
                    for bar in more_history {
                        if bar.complete {
                            history.push(bar);
                        }
                    }
                }
                //TODO Return on how many bars where asked for
                history.clone()
            } else {
                vec![]
            }
        }

        fn get_max_history(&self) -> Vec<Bar> {
            self.get_complete_bars(100000)
        }

        fn current_tick(&self) -> Tick {
            (*self.current_tick.lock().unwrap()).clone()
        }

        fn take_action(&self, action: &Action) -> ActionResult {
            match &self.mode {
                &Mode::Sim(ref slippage) => {
                    if let &Some(ref slippage) = slippage {
                        let mut rng = rand::thread_rng();
                        thread::sleep_ms(slippage.ind_sample(&mut rng));
                    }

                    match self.current_tick.lock() {
                        Ok(tick) => {
                            match action {
                                &Action::Buy => ActionResult::Price((*tick).ask),
                                &Action::Sell => ActionResult::Price((*tick).bid),
                                _ => ActionResult::None
                            }
                        },
                        Err(e) => ActionResult::None
                    }
                },
                &Mode::Live => {
                    if action != &Action::None {
                        let body = json!({
                            "order": {
                                "type": "MARKET",
                                "instrument": &self.instrument,
                                "units": if action == &Action::Buy { 100 } else { -100 }
                            }
                        }).to_string();
                        let client = Client::new();
                        match client.post(&self.order_url())
                            .header(header::Authorization(header::Bearer { token: self.token.clone() }))
                            .header(AcceptDatetimeFormat("UNIX".to_owned()))
                            .header(ContentType("application/json".to_owned()))
                            .body(body)
                            .send() {
                            Ok(mut res) => {
                                if res.status().is_success() {
                                    let body = res.json::<Value>();
                                    if let Ok(body) = body {
                                        //TODO More validation?
                                        if let Some(price) = body["orderFillTransaction"]["price"].parse_as_f64() {
                                            return ActionResult::Price(price);
                                        }
                                    }
                                }
                                ActionResult::None
                            },
                            Err(e) => ActionResult::None
                        }
                    } else {
                        ActionResult::None
                    }
                }
            }
        }

        fn get_open_trades(&self) -> (Vec<f64>, Action) {
            if let &Mode::Live = &self.mode {
                let client = Client::new();
                match client.get(&self.open_trades_url())
                    .header(header::Authorization(header::Bearer { token: self.token.clone() }))
                    .header(AcceptDatetimeFormat("UNIX".to_owned()))
                    .send() {
                    Ok(mut res) => {
                        if res.status().is_success() {
                            if let Ok(body) = res.json::<Value>() {
                                if let Some(trades) = body["trades"].as_array() {
                                    let mut ret: Vec<f64> = vec![];
                                    let mut action = Action::None;
                                    for trade in trades {
                                        if trade["instrument"].as_str().expect("Unable to parce trade instrument") == &self.instrument {
                                            ret.push(trade["price"].parse_as().expect("Unable to parse trade price"));
                                            let units = trade["initialUnits"].parse_as::<i32>().expect("Unable to parse trade units");
                                            if units > 0 {
                                                if action == Action::None {
                                                    action = Action::Buy
                                                } else if action != Action::Buy {
                                                    panic!("Expected all trades to be on the same side!")
                                                }
                                            } else {
                                                if action == Action::None {
                                                    action = Action::Sell
                                                } else if action != Action::Sell {
                                                    panic!("Expected all trades to be on the same side!")
                                                }
                                            }
                                        }
                                    }
                                    return (ret, action)
                                }
                            }
                        }
                    }
                    Err(e) => panic!("{:?}", e)
                }
                panic!("Failed to get list of open trades!")
            } else {
                (vec![], Action::None)
            }
        }

        fn granularity<'g, 'c: 'g>(&'c self) -> &'g Granularity {
            &self.granularity
        }

        fn reset(&self) {
            let mut history = self.history.lock().unwrap();
            history.clear();
        }
    }
}