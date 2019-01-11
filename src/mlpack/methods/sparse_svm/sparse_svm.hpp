/**
 * @file sparse_svm.hpp
 * @author Ayush Chamoli
 *
 * An implementation of Sparse SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

#include "sparse_svm_function.hpp"

namespace mlpack {
namespace svm {

class SparseSVM
{
 public:
  /**
   * Construct the SparseSVM class with the provided data and labels.
   * This will train the model.
   *
   * @tparam OptimizerType Desired differentiable separable optimizer
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param optimizer Desired optimizer.
   */
  template <typename OptimizerType = ens::ParallelSGD<>>
  SparseSVM(const arma::mat& data,
            const arma::Row<size_t>& labels,
            OptimizerType optimizer = OptimizerType());

  /**
   * Classify the given points, returning  predicted class label ({-1, 1})
   * for each data point.
   * The function calculates the (w * x + b) for a given data point. It then
   * chooses the class between {-1, 1} depending upon the result is positive
   * or negative.
   *
   * @param dataset Matrix of data points to be classified.
   * @param labels Predicted labels for each point.
   */
  void Classify(const arma::mat& dataset,
                arma::Row<size_t>& labels) const;

  /**
   * Computes accuracy of the learned model given the feature data and the
   * labels associated with each data point. Predictions are made using the
   * provided data and are compared with the actual labels.
   *
   * @param testData Matrix of data points using which predictions are made.
   * @param testLabels Vector of labels associated with the data.
   * @return Accuracy of the model.
   */
  double ComputeAccuracy(const arma::mat& testData,
                         const arma::Row<size_t>& testLabels) const;

  /**
   * Train the Sparse SVM with the given training data.
   *
   * @tparam OptimizerType Desired optimizer
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param optimizer Desired optimizer.
   * @return Objective value of the final point.
   */
  template <typename OptimizerType = ens::ParallelSGD<>>
  double Train(const arma::mat& data,
               const arma::Row<size_t>& labels,
               OptimizerType optimizer = OptimizerType());

  //! Get the model parameters.
  arma::mat& Parameters() { return parameters; }
  //! Get the model parameters.
  const arma::mat& Parameters() const { return parameters; }

  /**
   * Serialize the SparseSVM model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters);
  }

 private:
  //! Parameters after optimization.
  arma::mat parameters;
};

} // namespace regression
} // namespace mlpack

// Include implementation.
#include "sparse_svm.cpp"

#endif
