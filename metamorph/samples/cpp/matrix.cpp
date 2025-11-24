#include <iostream>
#include "../../probes/cpp_probe/probe.cpp"

int main() {
    int mat[2][2] = { {1, 2}, {3, 4} };

    std::string info = collectInfo(mat, typeid(mat).name());

    std::cout << "[C++ PROBE OUTPUT]\n";
    std::cout << info << std::endl;

    return 0;
}
