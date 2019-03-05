/**
 * @file copy.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the CopyTask class, which implements a generator of
 * instances of sequence copy task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_AUGMENTED_TASKS_COPY_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_COPY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

/**
 * Generator of instances of the binary sequence copy task.
 * The parameters are:
 * - maximum sequence length;
 * - number of sequence repetitions.
 * 
 * Input/output sequences are aligned to have the same length:
 * input sequence is padded with zeros from the right end,
 * output sequence is padded with zeros from the left end.
 * The sequences are formed of 2-dimensional vectors
 * of the format [sequence element, input flag],
 * where input flag = 0 iff first vector element is a sequence element.
 * 
 * Generated datasets are compliant with mlpack format -
 * every dataset element is shaped as a vector of
 * length (elem-length) * (input sequence length + target sequence length),
 * where elem-lemgth is 2 for input sequences and 1 for output sequences.
 * 
 * Example of generated dataset (sequence length = 3, repetition count = 2):
 * - Input sequence: [1,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1]
 * - Output sequences: [0,0,0,1,0,1,1,0,1]
 * 
 */
class CopyTask
{
 public:
  /**
   * Creates an instance of the sequence copy task.
   *
   * @param maxLength Maximum length of sequence
   *                  that has to be repeated by model.
   * @param nRepeats Number of repeats required to solve the task.
   * @param addSeparator Flag indicating whether generator
   *                     should emit separating symbol after input sequence.
   */
  CopyTask(const size_t maxLength,
           const size_t nRepeats,
           const bool addSeparator = false);
  /**
   * Generate dataset of a given size.
   *
   * @param input The variable to store input sequences.
   * @param labels The variable to store output sequences.
   * @param batchSize The dataset size.
   */
  void Generate(arma::field<arma::mat>& input,
                arma::field<arma::mat>& labels,
                const size_t batchSize,
                bool fixedLength = false) const;

  /**
   * Generate dataset of a given size and store it in
   * arma::mat object.
   * 
   * @param input The variable to store input sequences.
   * @param labels The variable to store output sequences.
   * @param batchSize The dataset size.
   */
  void Generate(arma::mat& input,
                arma::mat& labels,
                const size_t batchSize) const;

 private:
  // Maximum length of a sequence.
  size_t maxLength;
  // Number of repeats the model has to perform to complete the task.
  size_t nRepeats;
  // Flag indicating whether generator should produce
  // separator as part of the sequence
  bool addSeparator;
};

} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "copy_impl.hpp"

#endif
