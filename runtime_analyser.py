import sys
import time
import inspect
import types

class RuntimeAnalyzer:
    def __init__(self):
        self.call_stack = []
        self.data = []

    def trace_calls(self, frame, event, arg):
        if event == 'call':
            code = frame.f_code
            func_name = code.co_name
            filename = code.co_filename
            lineno = frame.f_lineno
            start_time = time.time()
            local_vars = frame.f_locals.copy()
            
            self.call_stack.append({
                'func_name': func_name,
                'filename': filename,
                'lineno': lineno,
                'start_time': start_time,
                'locals': local_vars
            })
            
        elif event == 'return':
            if self.call_stack:
                call_info = self.call_stack.pop()
                duration = time.time() - call_info['start_time']
                call_info.update({
                    'return_value': arg,
                    'duration': duration,
                    'return_type': type(arg).__name__
                })
                self.data.append(call_info)
        return self.trace_calls

    def analyze(self, target_module):
        sys.settrace(self.trace_calls)
        try:
            target_module.main()  # Expecting the target code to define a main()
        finally:
            sys.settrace(None)
        self.summarize()

    def summarize(self):
        print("\n=== Runtime Analysis Summary ===")
        for i, call in enumerate(self.data, 1):
            print(f"\n[{i}] Function: {call['func_name']}")
            print(f"File: {call['filename']} @ line {call['lineno']}")
            print(f"Duration: {call['duration']:.5f}s")
            print(f"Return Type: {call['return_type']}")
            print(f"Returned: {repr(call['return_value'])}")
            print(f"Locals at Call: {call['locals']}")
        print("\nTotal Functions Tracked:", len(self.data))

# Example usage:
if __name__ == "__main__":
    import example_target  # Your target program
    analyzer = RuntimeAnalyzer()
    analyzer.analyze(example_target)
