// Copyright PinaPL
//
// functions.cpp
// PinaPL
//
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>

double sigmoid(double x) {
    return (1/(1+exp(-x)));
}