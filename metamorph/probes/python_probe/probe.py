import inspect
import json
import time
import tracemalloc
import sys

def get_runtime_info(func, *args):
    info = {}

    # timing
    start = time.time()
    tracemalloc.start()

    result = func(*args)

    current, peak = tracemalloc.get_traced_memory()
    end = time.time()

    info["language"] = "Python"
    info["execution_time_ms"] = (end - start) * 1000
    info["memory_bytes_current"] = current
    info["memory_bytes_peak"] = peak

    info["type_info"] = {
        "function_name": func.__name__,
        "argument_types": [type(a).__name__ for a in args],
        "return_type": type(result).__name__
    }

    print(json.dumps(info, indent=2))
    return result

