import sys
sys.path.insert(0, "./worker")
import worker

def start_worker(source, name, models_loc, window, start, n_steps, test=False):
    new_worker = worker.Worker(source, name, models_loc, window, start, n_steps, test)
    new_worker.run()
