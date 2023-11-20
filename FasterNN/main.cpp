#include <iostream>
#include "Activation.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    std::vector<double> myVector = {0.4, 10.5, 2.3};

    double res = Sigmoid(myVector);
    return 0;
}
