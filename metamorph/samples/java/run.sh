#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SAMPLES_DIR="$ROOT_DIR/samples/java"
PROBE_DIR="$ROOT_DIR/probes/java_probe"

cd "$SAMPLES_DIR"

echo "[JAVA] Compiling sample & probe..."
javac "$SAMPLES_DIR/Matrix.java" "$PROBE_DIR/Probe.java"

echo "[JAVA] Running..."
java -cp "$SAMPLES_DIR:$PROBE_DIR" Matrix
