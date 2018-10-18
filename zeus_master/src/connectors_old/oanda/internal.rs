pub mod internal {
    use super::super::*;

    impl OANDA {
        pub fn open_trades_url(&self) -> String {
            format!("{}/v3/accounts/{}/openTrades", REST, &self.account)
        }

        pub fn order_url(&self) -> String {
            format!("{}/v3/accounts/{}/orders", REST, &self.account)
        }

        pub fn stream_url(&self) -> String {
            format!("{}/v3/accounts/{}/pricing/stream?instruments={}", STREAM, &self.account, &self.instrument)
        }

        pub fn get_history(token: &str, instrument: &str, granularity: &Granularity, bar_count: usize) -> Vec<Bar> {
            let url = format!("{}/v3/instruments/{}/candles?granularity={:?}&price=B&count={}&smooth=true", REST, instrument, granularity, bar_count);
            let client = Client::new();
            match client.get(&url)
                        .header(header::Authorization(header::Bearer { token: token.to_owned() }))
                        .header(AcceptDatetimeFormat("UNIX".to_owned()))
                        .send() {
                Ok(mut res) => {
                    if let Some(bars) = OANDA::parse_history(&mut res) {
                        return bars;
                    }
                }
                Err(e) => {}
            }
            vec![]
        }

        pub fn get_complete_bars(&self, bar_count: usize) -> Vec<Bar> {
            let client = Client::new();
            let mut epoch_seconds = chrono::Utc::now().timestamp();
            let mut bars: Vec<Bar> = Vec::new();
            while true {
                let len = bars.len();
                if len >= bar_count { break; }

                let mut need = bar_count - len;
                println!("{} bars still needed", need);
                need += 1;
                let uri = format!("{}/v3/instruments/{}/candles?to={}&granularity={:?}&price=B&count={}&smooth=true&includeLast=false",
                                  REST,
                                  &self.instrument,
                                  &epoch_seconds,
                                  &self.granularity,
                                  if need > 5000 { 5000 } else { need });
                println!("{}", &uri);
                match client.get(&uri)
                            .header(header::Authorization(header::Bearer { token: self.token.clone() }))
                            .header(AcceptDatetimeFormat("UNIX".to_owned()))
                            .header(ContentType("application/json".to_owned()))
                            .send() {
                    Ok(mut res) => {
                        if let Some(mut res_bars) = OANDA::parse_history(&mut res) {
                            res_bars.reverse();
                            for bar in res_bars {
                                epoch_seconds = bar.date;
                                if bar.complete {
                                    bars.push(bar);
                                }
                            }
                        }
                        else {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        break;
                    }
                }
            }
            bars.reverse();
            while bars.len() > bar_count {
                bars.remove(0);
            }
            bars
        }

        fn parse_history(res: &mut Response) -> Option<Vec<Bar>> {
            if res.status().is_success() {
                if let Ok(value) = res.json::<Value>() {
                    //TODO Verify symbol and granularity?
                    if let Some(candles) = value["candles"].as_array() {
                        if candles.len() > 0 {
                            return Some(candles.iter().map(|candle| {
                                let bid = &candle["bid"];
                                Bar {
                                    open: bid["o"].parse_as_f64().unwrap(),
                                    high: bid["h"].parse_as_f64().unwrap(),
                                    low: bid["l"].parse_as_f64().unwrap(),
                                    close: bid["c"].parse_as_f64().unwrap(),
                                    volume: candle["volume"].as_i64().unwrap() as i32,
                                    complete: candle["complete"].as_bool().unwrap(),
                                    date: candle["time"].parse_as_f64().unwrap() as i64
                                }
                            }).collect());
                        }
                    }
                }
            }
            None
        }


        pub fn receive_ticks(&self) {
            let url = self.stream_url();
            let current_tick = Arc::clone(&self.current_tick);
            let token = self.token.clone();
            thread::spawn(move || {
                let client = Client::new();
                loop {
                    if let Ok(res) = client.get(&url).header(header::Authorization(header::Bearer { token: token.clone() })).send() {
                        if let StatusCode::Ok = res.status() {
                            let mut data: [u8; 1024] = [0; 1024];
                            let mut reader = BufReader::new(res);
                            while let Ok(read) = reader.read(&mut data) {
                                if let Ok(value) = json::from_slice::<Value>(&data[0..read]) {
                                    if let Some(tick) = OANDA::parse_tick(value) {
                                        //TODO THIS IS A HACK....
                                        //println!("New tick received: {:?}", tick);
                                        if let Ok(mut current_tick) = current_tick.lock() {
                                            *current_tick = tick;
                                        }
                                    }
                                }
                            }
                        };
                    };
                }
            });
        }

        pub fn parse_tick(val: Value) -> Option<Tick> {
            //TODO Add `status` and `instrument` checks
            if let Some(val_type) = val["type"].as_str() {
                if val_type == "PRICE" {
                    if let Some(bid) = val["bids"][0]["price"].parse_as_f64() {
                        if let Some(ask) = val["asks"][0]["price"].parse_as_f64() {
                            if let Some(timestamp) = val["time"].parse_as_f64() {
                                return Some(Tick { bid, ask, timestamp: timestamp as i64 });
                            }
                        }
                    }
                }
            }
            None
        }
    }
}