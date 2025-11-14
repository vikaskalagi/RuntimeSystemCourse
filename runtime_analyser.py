import sys
import time
import json
import threading
import inspect
import types
import gc

class RuntimeProbe:
    def __init__(self):
        self.events = []

    def trace_calls(self, frame, event, arg):
        if event not in ('call', 'return'):
            return self.trace_calls

        code = frame.f_code
        func_name = code.co_name
        lineno = frame.f_lineno
        filename = frame.f_filename
        call_depth = len(inspect.stack(0)) - 1  # approximate call depth

        if event == 'call':
            start_time = time.time()
            local_vars = {k: type(v).__name__ for k, v in frame.f_locals.items()}
            frame.f_locals['_probe_start_time'] = start_time
            frame.f_locals['_probe_locals_snapshot'] = local_vars

        elif event == 'return':
            start_time = frame.f_locals.get('_probe_start_time', time.time())
            duration = (time.time() - start_time) * 1000  # ms
            locals_snapshot = frame.f_locals.get('_probe_locals_snapshot', {})
            ret_val = arg
            ret_type = type(ret_val).__name__

            event_data = {
                "timestamp": time.time(),
                "language": "Python",
                "execution": {
                    "function": func_name,
                    "call_depth": call_depth,
                },
                "memory": {
                    "object_size": sys.getsizeof(ret_val),
                    "stack_frame_bytes": sys.getsizeof(frame.f_locals),
                },
                "type": {
                    "type_name": type(ret_val).__name__,
                    "is_class": isinstance(ret_val, type),
                },
                "dynamic": {
                    "locals": locals_snapshot,
                    "return_type": ret_type
                },
                "performance": {
                    "exec_time_ms": duration
                },
                "gc": {
                    "enabled": True,
                    "collected": gc.collect(),
                    "gc_objects": len(gc.get_objects())
                }
            }
            self.events.append(event_data)

        return self.trace_calls

    def analyze(self, target_module):
        sys.settrace(self.trace_calls)
        try:
            target_module.main()
        finally:
            sys.settrace(None)
        self.output_results()

    def output_results(self):
        print(json.dumps(self.events, indent=2))

# Example usage
if __name__ == "__main__":
    import matrix_mul  # Replace with your target file/module
    probe = RuntimeProbe()
    probe.analyze(matrix_mul)
