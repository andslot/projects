//
// Created by slotk on 20/11/2023.
//

#ifndef FASTERNN_ACTIVATION_H
#define FASTERNN_ACTIVATION_H

#include <vector> // include vector

std::vector<double> Softmax(const std::vector<double>& z);
std::vector<double> Linear(const std::vector<double>& z);
double Sigmoid(const std::vector<double>& z);
double ReLU(const std::vector<double>& z);
// More activation functions to come

#endif //FASTERNN_ACTIVATION_H
