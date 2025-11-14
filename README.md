# Metamorph - A toolkit that lets us inspect and compare runtime information
## F25 Runtime Systems course project

Team mates:
- Bhargavi Kurukunda
- Vikas Kalagi

## What the Runtime System Info includes:
### 1. Execution Model

What the runtime needs to run your program:
* Call stack info (current function, stack depth)
* Control flow transitions (function calls, returns)
* Thread state (running, blocked, waiting, suspended)
* Scheduling strategy (preemptive, cooperative, interpreter loop)

Why it matters:
* This shows how each language executes code and manages execution state.

### 2. Memory Model

The runtime’s biggest job: managing memory.

We should collect:
* Heap usage (current size, growth)
* Stack frame size
* Object layout (fields, offsets, type tags)
* Pointer/reference behaviour
* Memory fragmentations (if exposed)
* Garbage collection presence (Java, Python)
* Manual memory mgmt (C++)

Why it matters:
* This is where languages differ the most — important for cross-language comparison.

### 3. Type System Metadata

Record:

* Type name
* Type category (class, object, primitive, pointer)
* Method list & signatures
* Field list
* Attributes/modifiers (public/private, static/final)
* Inheritance hierarchy
* Interfaces/abstract classes
* Dynamic type info (RTTI)

### 4. Dynamic Behaviour

What happens while the program is running:

* Actual types of objects at runtime
* Dynamic dispatch / virtual functions
* Method override resolution
* Polymorphic calls
* Object creation frequency
* Mutation of objects (Python: allowed; Java: structured; C++: raw pointer-based)

Why it matters:
* This shows how different languages resolve behaviour dynamically.

### 5. Garbage Collector (if any)

Applicable to Python & Java.

We can capture:

* GC type 
* GC triggers
* GC pauses/duration

Why it matters:

Massive difference between Python (refcounting + cycle detector) and Java (generational, concurrent collectors).
You can compare both with C++ (no GC).

### 6. Performance Metrics

Collect real runtime stats:

* Function execution time
* Object allocation time
* Number of function calls
* Memory growth over time
* Thread scheduling delays

### 7. Error & Exception Model

At runtime, you can collect:

* What exceptions occurred
* Where they were thrown
* Uncaught vs caught errors
* Stack-unwinding behaviour

Why it matters:

Huge differences:
C++: stack unwinding, no unified exception type.
Java: mandatory checked exceptions.
Python: fully dynamic exception hierarchy.
