use std::io::{Write, stdout};

use chrono::{Utc, Datelike, Timelike, DateTime, TimeZone};
use ansi_term;
use num::Num;
use tensorflow as tf;

use connectors::Connector;
use data::models::*;
use data::utils::*;
use super::{Action, ActionResult};

mod utils;
use self::utils::*;

pub mod backend;
pub mod bar;
pub mod management;

use self::backend::Backend;
use self::backend::tensorflow::Tensorflow;
use self::backend::tensorflow::resolvers::*;
use self::management::*;

pub trait Trader<'c, C: 'c + Connector, I, B: Backend<I>, T: TradeManager> {
    fn epoch(&self) -> usize;
    fn backend<'b, 's: 'b>(&'s mut self) -> &'b mut B;
    fn compile_input(&mut self, bar: Bar) -> Option<I>;
    fn evaluate_trades(&mut self, connector: &'c C, trade_manager: &mut T);
    fn display(&self, epoch: usize, current_tick: &Tick) -> String { String::from("          ") }

    fn evaluate(&mut self, connector: &'c C, mut trade_manager: &mut T, bar: Bar) -> String {
        {
            let timestamp = bar.date;
            if let Some(input) = self.compile_input(bar) {
                let action = self.backend().evaluate(&input); //TODO Maybe the backend should be called by the Args or Features in order to determine the action
                trade_manager.take_action(connector, action, timestamp);
            }
        }
        self.evaluate_trades(connector, &mut trade_manager);

        let epoch = self.epoch();
        let tick = connector.current_tick();
        let metrics = trade_manager.display(epoch, &tick);
        print!("\r{}{}", &metrics, self.display(epoch, &tick));
        stdout().flush();
        metrics
    }
}