import worker
import argparse

def start_worker(source, name, models_loc, window, n_steps):
    this_worker = worker.Worker(source, name, models_loc, window, n_steps)
    this_worker.run()
