from ctypes import *
from .data import *
import time
import sys
import codecs

def slashescape(err):
    """ codecs error handler. err is UnicodeDecode instance. return
    a tuple with a replacement for the unencodable part of the input
    and a position where encoding should continue"""
    #print err, dir(err), err.start, err.end, err.object[:err.start]
    thebyte = err.object[err.start:err.end]
    repl = u'\\x'+hex(ord(thebyte))[2:]
    return (repl, err.end)

codecs.register_error('slashescape', slashescape)

class Zeus:
    def __init__(self, instrument, granularity, live=False, margin=None):
        self.instrument = instrument;
        self.granularity = granularity;

        self.lib = windll.LoadLibrary('target\debug\zeus.dll')

        if live == False:
            if margin is not None:
                self.lib.create_dev_session.argtypes = [POINTER(RawString), c_char, c_int, c_double]
                self.lib.create_dev_session.restype = c_void_p
                self.sess = self.lib.create_dev_session(as_raw_str(instrument), c_char(bytes(granularity[0], 'utf-8')), int(granularity[1:]), c_double(margin))
            else:
                self.lib.create_session.argtypes = [POINTER(RawString), c_char, c_int]
                self.lib.create_session.restype = c_void_p
                self.sess = self.lib.create_session(as_raw_str(instrument), c_char(bytes(granularity[0], 'utf-8')), int(granularity[1:]))
        else:
            self.lib.create_live_session.argtypes = [POINTER(RawString), c_char, c_int]
            self.lib.create_live_session.restype = c_void_p
            self.sess = self.lib.create_live_session(as_raw_str(instrument), c_char(bytes(granularity[0], 'utf-8')), int(granularity[1:]))

        #  Data
        self.lib.load_history.argtypes = [c_void_p, c_long, c_long, c_void_p]
        self.lib.load_history.restype = c_void_p
        self.lib.stream_bars.argtypes = [c_void_p, c_bool, c_size_t, c_void_p]
        self.lib.stream_bars.restype = c_void_p
        self.lib.stream_range.argtypes = [c_void_p, c_bool, c_long, c_long, c_void_p]
        self.lib.stream_range.restype = c_void_p
        self.lib.stream_live.argtypes = [c_void_p, c_void_p]
        self.lib.stream_live.restype = c_void_p

        #  Broker Info
        self.lib.current_balance.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.current_balance.restype = c_void_p
        self.lib.current_price.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.current_price.restype = c_void_p
        self.lib.unrealized_pl.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.unrealized_pl.restype = c_void_p
        self.lib.unrealized_balance.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.unrealized_balance.restype = c_void_p
        self.lib.percent_change.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.percent_change.restype = c_void_p
        self.lib.used_margin.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.used_margin.restype = c_void_p
        self.lib.available_margin.argtypes = [c_void_p, POINTER(c_double)]
        self.lib.available_margin.restype = c_void_p
        self.lib.position_size.argtypes = [c_void_p, POINTER(c_int)]
        self.lib.position_size.restype = c_void_p
        self.lib.units_available.argtypes = [c_void_p, POINTER(c_uint)]
        self.lib.units_available.restype = c_void_p
        self.lib.stats.argtypes = [c_void_p]
        self.lib.stats.restype = c_void_p
        # TODO self.lib.unrealized_trade_pl.restype = c_void_p

        #  Trading
        self.lib.place_trade.argtypes = [c_void_p, POINTER(TradeRequest), POINTER(RawString)]
        self.lib.place_trade.restype = c_void_p
        self.lib.close_trade.argtypes = [c_void_p, POINTER(RawString), POINTER(c_double)]
        self.lib.close_trade.restype = c_void_p
        self.lib.close_units.argtypes = [c_void_p, c_uint, POINTER(c_double)]
        self.lib.close_units.restype = c_void_p

        # self.lib.test.argtypes = [c_void_p]
        # self.lib.test.restype = None

    def stats(self):
        self.sess = self.lib.stats(self.sess)

    def _get_value(self, func):
        temp = c_double(0.0)
        self.sess = func(self.sess, byref(temp))
        return temp.value

    def current_balance(self):
        return self._get_value(self.lib.current_balance)

    def unrealized_pl(self):
        return self._get_value(self.lib.unrealized_pl)

    def unrealized_balance(self):
        return self._get_value(self.lib.unrealized_balance)

    def percent_change(self):
        return self._get_value(self.lib.percent_change)

    def used_margin(self):
        return self._get_value(self.lib.used_margin)

    def available_margin(self):
        return self._get_value(self.lib.available_margin)

    def position_size(self):
        temp = c_int(0)
        self.sess = self.lib.position_size(self.sess, byref(temp))
        return temp.value

    def units_available(self):
        temp = c_uint(0)
        self.sess = self.lib.units_available(self.sess, byref(temp))
        return temp.value

    def load_history(self, from_epoch, to_epoch):
        history = History()
        self.sess = self.lib.load_history(self.sess, from_epoch, to_epoch, byref(history))
        history

    def stream_bars(self, count, callback, file=False):
        def on_bar(sess, bar):
            self.sess = sess
            callback(bar.contents)
            sys.stdout.flush()
            return self.sess
        BARFUNC = WINFUNCTYPE(c_void_p, c_void_p, POINTER(Bar))(on_bar)
        self.sess = self.lib.stream_bars(self.sess, file, count, byref(BARFUNC))

    def stream_range(self, from_epoch, to_epoch, callback, file=False):
        def on_bar(sess, bar):
            self.sess = sess
            callback(bar.contents)
            sys.stdout.flush()
            return self.sess
        BARFUNC = WINFUNCTYPE(c_void_p, c_void_p, POINTER(Bar))(on_bar)
        self.sess = self.lib.stream_range(self.sess, file, from_epoch, to_epoch, byref(BARFUNC))

    def stream_live(self, callback):
        def on_bar(sess, bar):
            self.sess = sess
            contents = None
            try:
                contents = bar.contents
            except:
                pass
            if contents is not None:
                callback(bar.contents)
            sys.stdout.flush()
            return self.sess
        BARFUNC = WINFUNCTYPE(c_void_p, c_void_p, POINTER(Bar))(on_bar)
        self.sess = self.lib.stream_live(self.sess, byref(BARFUNC))

    def place_trade(self, quantity, pos):
        req = TradeRequest(as_raw_str(self.instrument), quantity, as_raw_str(pos))
        id = as_raw_str("")
        self.sess = self.lib.place_trade(self.sess, byref(req), byref(id))
        return id.raw.decode('utf-8', 'ignore')[:id.len]

    def close_trade(self, id):
        id = as_raw_str(id)
        var = c_double(0.0)
        self.sess = self.lib.close_trade(self.sess, byref(id), byref(var))
        return var.value

    def close_units(self, units):
        var = c_double(0.0)
        self.sess = self.lib.close_units(self.sess, units, byref(var))
        return var.value

    def test(self):
        self.lib.test(self.sess)
