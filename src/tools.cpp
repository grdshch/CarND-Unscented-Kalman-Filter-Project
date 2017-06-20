#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse;
  if (estimations.empty() || estimations.size() != ground_truth.size()) {
    std::cout << "Error: estimations and ground_truth must have equal sizes" << std::endl;
    return rmse;
  }
  rmse = VectorXd::Zero(estimations[0].size());

  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd error = estimations[i] - ground_truth[i];
    error = error.array() * error.array();
    rmse += error;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();
  return rmse;
}