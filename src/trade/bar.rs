use super::*;
use features::{Bars, Features};
use std::marker::PhantomData;

pub struct BarTrader<F: Features, I, B: Backend<I>> {
    bars: Bars<F>,
    backend: B,
    backend_type: PhantomData<I>
}

impl<F: Features, I, B: Backend<I>> BarTrader<F, I, B> {
    pub fn new<'c, C: 'c + Connector>(bars: Bars<F>, backend: B, connector: &'c C) -> Self {
        panic_if_trades_open(connector);
        //let bars = connector.get_history(args.sample_size);
        BarTrader {
            bars,
            backend,
            backend_type: PhantomData
        }
    }
}

impl<'c, C: 'c + Connector, F: Features, I: Convert<Vec<Vec<f64>>>, B: Backend<I>, T: TradeManager> Trader<'c, C, I, B, T> for BarTrader<F, I, B> {

    fn epoch(&self) -> usize {
        self.bars.len()
    }

    fn backend<'b, 's: 'b>(&'s mut self) -> &'b mut B {
        &mut self.backend
    }

    fn compile_input(&mut self, bar: Bar) -> Option<I> {
        self.bars.compile_input(bar).map(|sample| I::convert(sample))
    }

    fn evaluate_trades(&mut self, connector: &'c C, trade_manager: &mut T) {
        let tick = connector.current_tick();
        trade_manager.evaluate(connector, |trade| {
            self.bars.features().should_close(trade, &tick)
        });
    }
}