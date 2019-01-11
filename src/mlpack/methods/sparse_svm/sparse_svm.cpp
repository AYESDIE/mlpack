/**
 * @file sparse_svm.cpp
 * @author Ayush Chamoli
 *
 * Implementation of Sparse SVM.
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

void SparseSVM::Classify(const arma::mat& dataset,
                         arma::Row<size_t>& labels)
    const
{
  labels = arma::sign(parameters.cols(dataset.n_elem) * dataset +
                      parameters.tail_cols(dataset.n_elem));
}

double SparseSVM::ComputeAccuracy(const arma::mat& testData,
                                  const arma::Row<size_t>& testLabels)
    const
{
  arma::Row<size_t> labels;

  // Get predictions for the provided data.
  Classify(testData, labels);

  // Increment count for every correctly predicted label.
  size_t count = 0;
  for (size_t i = 0; i < labels.n_elem ; i++)
    if (testLabels(i))
      count++;

  // Return percentage accuracy
  return (count * 100.0) / labels.n_elem;
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