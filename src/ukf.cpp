#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  Sigma_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  radar_out.open("radar_nis.txt", fstream::out);
  laser_out.open("laser_nis.txt", fstream::out);

  is_initialized_ = false;
}

UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      // convert measurement from polar to cartesian coordinates
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  // Prediction
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_ ||
      meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
    return;
  }

  // time from previous measurement in seconds
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  // Updating
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else if (use_laser_)
  {
    UpdateLidar(meas_package);
  }
}
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_ * std_a_, 0,
       0, std_yawdd_ * std_yawdd_;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;

  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  //create sigma point matrix
  MatrixXd Sigma = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();
  A = A * sqrt(lambda_ + n_aug_);

  Sigma = x_aug.rowwise().replicate(2 * n_aug_ + 1);  // set each column to x_aug vector
  Sigma.block(0, 1, n_aug_, n_aug_) += A;  // add matrix A to column block [1, n_aug]
  Sigma.block(0, n_aug_ + 1, n_aug_, n_aug_) -= A;  // subtract matrix A from column block [n_aug + 1, 2 * n_aug + 1]

  for (int i = 0; i < n_aug_ * 2 + 1; ++i) {
    VectorXd x = Sigma.col(i);
    double v = x(2);
    double yaw = x(3);
    double yawd = x(4);
    double nua = x(5);
    double nup = x(6);

    if (fabs(yawd) < 0.0001) {
      x(0) += v * cos(yaw) * delta_t;
      x(1) += v * sin(yaw) * delta_t;
    }
    else {
      x(0) += v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      x(1) += v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }

    x(0) += 0.5 * nua * delta_t * delta_t * cos(yaw);
    x(1) += 0.5 * nua * delta_t * delta_t * sin(yaw);
    x(2) += delta_t * nua;
    x(3) += yawd * delta_t + 0.5 * nup * delta_t * delta_t;
    x(4) += delta_t * nup;

    Sigma_pred_.col(i) = x.head(n_x_);
  }

  //predict state mean
  x_ = Sigma_pred_ * weights_;

  //predict state covariance matrix
  P_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diff = Sigma_pred_.col(i) - x_;
    //angle normalization
    while (diff(3) > M_PI) {
      diff(3) -= 2. * M_PI;
    }
    while (diff(3) < -M_PI) {
      diff(3) += 2. * M_PI;
    }
    P_ += weights_(i) * diff * diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  //transform sigma points into measurement space
  Zsig = Sigma_pred_.block(0, 0, 2, 2 * n_aug_ + 1);

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diff = Zsig.col(i) - z_pred;
    S += weights_(i) * diff * diff.transpose();
  }
  S(0, 0) += std_laspx_ * std_laspx_;
  S(1, 1) += std_laspy_ * std_laspy_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Tc += weights_(i) * (Sigma_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  P_ -= K * S * K.transpose();

  double nis = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() * (meas_package.raw_measurements_ - z_pred);
  laser_out << nis << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x = Sigma_pred_.col(i);
    Zsig.col(i)(0) = sqrt(x(0) * x(0) + x(1) * x(1));
    Zsig.col(i)(1) = atan2(x(1), x(0));
    Zsig.col(i)(2) = (x(0) * cos(x(3)) * x(2) + x(1) * sin(x(3)) * x(2)) / sqrt(x(0) * x(0) + x(1) * x(1));
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diff = Zsig.col(i) - z_pred;
    S += weights_(i) * diff * diff.transpose();
  }
  S(0, 0) += std_radr_ * std_radr_;
  S(1, 1) += std_radphi_ * std_radphi_;
  S(2, 2) += std_radrd_ * std_radrd_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Tc += weights_(i) * (Sigma_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  P_ -= K * S * K.transpose();

  double nis = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() * (meas_package.raw_measurements_ - z_pred);
  radar_out << nis << std::endl;
}
