//
// Created by slotk on 20/11/2023.
//

#include <stdexcept>
#include "Activation.h"

class NotImplemented : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override {
        return "This functionality have not been implemented yet.";
    }
};

std::vector<double> Softmax(const std::vector<double>& z) {
    throw NotImplemented();
}

std::vector<double> Linear(const std::vector<double>& z) {
    throw NotImplemented();
}

double Sigmoid(const std::vector<double>& z) {
    throw NotImplemented();
}

double ReLU(const std::vector<double>& z) {
    throw NotImplemented();
}