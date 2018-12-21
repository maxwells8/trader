import sys
sys.path.insert(0, "./worker")
from worker import Worker

def start_worker(name, instrument, granularity, models_loc, start, test=False):
    new_worker = Worker(name, instrument, granularity, models_loc, start, test)
    new_worker.run()
