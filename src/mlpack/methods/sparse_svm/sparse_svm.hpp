/**
 * @file sparse_svm.hpp
 * @author Ayush Chamoli
 *
 * An implementation of softmax regression.
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
 * Classify the given points, returning class probabilities for each point.
 *
 * @param dataset Matrix of data points to be classified.
 * @param probabilities Class probabilities for each point.
 */
  void Classify(const arma::mat& dataset,
                arma::mat& probabilities) const;

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

  //! Gets the features size of the training data
  size_t FeatureSize() const { return parameters.n_cols; }

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
