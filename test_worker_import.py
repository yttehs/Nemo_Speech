"""
Checks whether a spawned worker process (same pattern the simulator itself
uses: set_start_method('spawn') + ProcessPoolExecutor) resolves the
data_simulation import to the same file as the main process does.

Run from your repo root:
    python test_worker_import.py
"""
import concurrent.futures
from multiprocessing import set_start_method


def worker_check(_):
    from nemo.collections.asr.data import data_simulation
    return data_simulation.__file__


if __name__ == "__main__":
    from nemo.collections.asr.data import data_simulation
    print(f"Main process imports from:   {data_simulation.__file__}")

    set_start_method('spawn', force=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as tp:
        futures = [tp.submit(worker_check, i) for i in range(2)]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            print(f"Worker {i} imports from:       {f.result()}")
