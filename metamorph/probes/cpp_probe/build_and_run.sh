#!/bin/bash
set -e
g++ samples/cpp/matrix.cpp -std=c++17 -ljsoncpp -o matrix
./matrix
