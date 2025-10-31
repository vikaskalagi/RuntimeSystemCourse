* Background research on runtimes of different languages.
Week 5 project progress
-------------
# 📘 MetaMorph: Using Insights from "Classport — Designing Runtime Dependency Introspection for Java"

## 🧩 Paper Overview

**Reference:**  
*Cofano, G., Zuddas, D., & Montresor, A. (2025).*  
**“Classport: Designing Runtime Dependency Introspection for Java.”**  
Available at [arXiv:2510.20340](https://arxiv.org/pdf/2510.20340)

---

### 🔍 Key Insights from the Paper

1. **Problem Identified**
   - Java provides runtime reflection (via `java.lang.reflect`) for inspecting classes, methods, and fields.
   - However, it lacks built-in mechanisms to introspect **runtime dependency usage** — i.e., which libraries or modules are actually *used* during execution.

2. **Proposed Solution: Classport**
   - Introduces two main components:
     - **Embedder:** A build-time tool that injects dependency metadata (group, artifact, version) into class files using annotations.
     - **Introspector:** A runtime Java agent that detects which annotated classes are loaded or executed.
   - Together, these components enable **runtime dependency introspection** — identifying dependencies that are actually used, not just declared.

3. **Evaluation Results**
   - Tested on six open-source Java projects.
   - Metadata embedding adds minimal size overhead.
   - Runtime monitoring works effectively with low performance impact.
   - Accuracy depends on the workload — unused classes remain invisible.

4. **Use Cases Highlighted**
   - Detecting unused dependencies.
   - Tracking vulnerability exposure at runtime.
   - Enforcing security rules based on dependency usage.

---

## 🧠 How MetaMorph Builds on These Insights

### 1. Metadata Embedding and Extraction
- **Inspired by Classport’s architecture**, MetaMorph adopts a similar concept:
  - For **Java**, build-time metadata embedding + runtime introspection via reflection APIs.
  - For **C++**, compile-time metadata (using RTTI or external reflection libraries).
  - For **Python**, runtime metadata extraction via the `inspect` module.
- All metadata is normalized into a common **JSON schema**, enabling cross-language comparison.

### 2. Workload and Coverage
- Classport notes that introspection accuracy depends on which code paths are executed.
- **Design implication:** MetaMorph’s example programs (e.g., `Person`, `Student`, `Book`) are intentionally designed to exercise various language features — inheritance, methods, and runtime mutations — ensuring complete metadata coverage.

### 3. Overhead and Feasibility
- Classport’s lightweight embedding demonstrates that runtime introspection is practical.
- MetaMorph focuses on small-scale examples to stay within course constraints, emphasizing **clarity over performance**.

### 4. Expanding the Use Case
- MetaMorph generalizes the concept: instead of only tracking dependencies, it visualizes **runtime object metadata** across three languages.
- Example: For the same conceptual structure (`Person`), compare how Java, Python, and C++ represent:
  - Type name
  - Field list
  - Method count
  - Memory model (static vs dynamic)

### 5. Architectural Inspiration
- MetaMorph reuses Classport’s two-phase model:
  - **Phase 1: Metadata Collection**
    - Java agent, Python script, or C++ reflection library collects runtime data.
  - **Phase 2: Visualization**
    - Python-based dashboard aggregates and displays metadata side-by-side.
- This modular approach simplifies implementation and testing for each language separately.

### 6. Comparative Study Opportunity
- Classport focuses on **runtime dependencies in Java**.  
- MetaMorph extends this idea to a **cross-language reflection comparison**:
  - Java → rich managed reflection, limited runtime mutation.
  - Python → dynamic runtime mutation, easy introspection.
  - C++ → limited built-in reflection, relies on RTTI or libraries.
- This comparative analysis highlights trade-offs in runtime system design.

---

## 🧱 Implementation Alignment

| Language | Technique | Metadata Extraction | Notes |
|-----------|------------|--------------------|-------|
| **Python** | `inspect`, `sys`, `type()` | Runtime object metadata | No build step needed |
| **Java** | Reflection API, optional agent | Class/method metadata | Can follow Classport-style embedding |
| **C++** | `typeid`, RTTI, external libs (RTTR/meta) | Compile-time + runtime metadata | Limited dynamic reflection |

---

## 💡 Summary

**Classport’s core contribution** — embedding metadata at build time and retrieving it dynamically — validates that runtime introspection is feasible in managed environments like Java.  
**MetaMorph extends this idea** by applying similar principles across three languages, aiming to unify their runtime representations and make reflection truly cross-language.

---
