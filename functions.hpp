// Copyright PinaPL
//
// functions.hpp
// PinaPL
//

#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>

double sigmoid(double x);
double sigmoid_derivative(double x);
double tanh_derivative(double x);
double tanh2(double x);
double tanhyp(double x);

#endif
