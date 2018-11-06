use super::*;

pub struct FIFO {
    open: Option<Trade>,
    closed: Vec<Trade>,
    reverse_position: bool
}

impl FIFO {
    pub fn new(reverse_position: bool) -> Self {
        FIFO { open: None, closed: vec![], reverse_position }
    }

    fn can_take_action(&self, action: &Action) -> bool {
        self.open.as_ref().map_or_else(|| true, |trade| &trade.action != action)
    }
}

impl TradeManager for FIFO {

    fn open<'t, 's: 't>(&'s self) -> Vec<&'t Trade> {
        if self.open.is_some() {
            vec![self.open.as_ref().unwrap()]
        }
        else {
            vec![]
        }
    }

    fn closed<'t, 's: 't>(&'s self) -> Vec<&'t Trade> {
        let mut ret: Vec<&Trade> = vec![];
        for closed in &self.closed {
            ret.push(closed)
        }
        ret
    }

    fn take_action<'c, C: 'c + Connector>(&mut self, connector: &'c C, action: Action, timestamp: i64) {
        if self.can_take_action(&action) {
            match connector.take_action(&action) {
                ActionResult::Price(price) => {
                    if self.open.is_none() {
                        self.open = Some(Trade {
                            open: price,
                            close: None,
                            timestamp,
                            action,
                        });
                    }
                    else {
                        let mut trade = self.open.take().unwrap();
                        trade.close = Some(price);
                        self.closed.push(trade);

                        if self.reverse_position {
                            self.take_action(connector, action, timestamp);
                        }
                    }
                }
                _ => {} //TODO Better Error Handling
            }
        }
    }

    fn evaluate<'c, C: 'c + Connector, F>(&mut self, connector: &'c C, should_close: F) where F: Fn(&Trade) -> bool {
        if self.open.is_some() {
            if should_close(self.open.as_ref().unwrap()) {
                let mut trade = self.open.take().unwrap();
                match connector.take_action(&trade.action) {
                    ActionResult::Price(price) => {
                        trade.close = Some(price);
                        self.closed.push(trade);
                    }
                    _ => {
                        self.open = Some(trade);
                    }
                }
            }
        }
    }
}