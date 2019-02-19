import os
import sys
sys.path.insert(0, "./worker")
from live_worker import Worker

instrument = "EUR_USD"
granularity = "M1"
dir_path = os.path.dirname(os.path.realpath(__file__))
models_loc = dir_path + '/models/'

new_worker = Worker(instrument, granularity, models_loc)
new_worker.run()
