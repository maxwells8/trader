from ctypes import *
from zeus_master.python.wrapper.zeus import Zeus as Z

def do(bar):
    global zeus
    id = zeus.place_trade(1000, "Long")
    print(zeus.unrealized_pl())
    print(zeus.close_trade(id))
    print(zeus.percent_change())
    print()

zeus = Z("EUR_USD", "M1")
zeus.stream_bars(10, do)
print(zeus.unrealized_balance())
