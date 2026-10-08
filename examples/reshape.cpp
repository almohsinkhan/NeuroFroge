#include "../include/tensor.h"
#include <iostream>

int main() {

    Tensor a({2, 3});

    for (int i = 0; i < a.size(); i++)
        a.flat(i) = i + 1;

    Tensor b = a.reshape({3, 2});

    std::cout << "Original:\n";

    for (int i = 0; i < a.size(); i++) {
        std::cout << a.flat(i) << " ";

        if ((i + 1) % 3 == 0)
            std::cout << "\n";
    }

    std::cout << "\nReshaped:\n";

    for (int i = 0; i < b.size(); i++) {
        std::cout << b.flat(i) << " ";

        if ((i + 1) % 2 == 0)
            std::cout << "\n";
    }

    b.flat(1) = 999;

std::cout << "\nAfter modifying reshaped tensor:\n";

for (int i = 0; i < a.size(); i++) {
    std::cout << a.flat(i) << " ";

    if ((i + 1) % 3 == 0)
        std::cout << "\n";
}

Tensor c = a.reshape({-1, 2});
c.printShape();

Tensor d = a.reshape({2, -1});
d.printShape();
    return 0;
}