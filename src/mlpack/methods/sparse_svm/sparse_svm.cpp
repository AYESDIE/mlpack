/**
 * @file softmax_regression_impl.hpp
 * @author Ayush Chamoli
 *
 * Implementation of softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_CPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_CPP

// In case it hasn't been included yet.
#include "sparse_svm.hpp"

namespace mlpack {
namespace svm {

template <typename OptimizerType>
SparseSVM::SparseSVM(
    const arma::mat& data,
    const arma::Row<size_t>& labels,
    OptimizerType optimizer)
{
  Train(data, labels, optimizer);
}

void SparseSVM::Classify(const arma::mat &dataset,
                         arma::mat &probabilities)
    const
{
  if (dataset.n_rows != FeatureSize())
  {
    std::ostringstream oss;
    oss << "SparseSVM::Classify(): dataset has " << dataset.n_rows
        << " dimensions, but model has " << FeatureSize() << " dimensions!";
    throw std::invalid_argument(oss.str());
  }

  /* To be made */
}

template <typename OptimizerType>
double SparseSVM::Train(const arma::mat& data,
             const arma::Row<size_t>& labels,
             OptimizerType optimizer)
{
  SparseSVMFunction svm(data, labels);
  if (parameters.is_empty())
    parameters = svm.InitialPoint();

  // Train the model.
  Timer::Start("sparse_svm_optimization");
  const double out = optimizer.Optimize(svm, parameters);
  Timer::Stop("sparse_svm_optimization");

  Log::Info << "SparseSVM::SparseSVM(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

} // namespace regression
} // namespace mlpack

#endif