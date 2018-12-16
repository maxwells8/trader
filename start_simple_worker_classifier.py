import sys
sys.path.insert(0, "./worker")
import simple_worker_classifier

def start_worker(instrument, granularity, start, n_steps):
    new_worker = simple_worker_classifier.Worker(instrument, granularity, start, n_steps)
    new_worker.run()
