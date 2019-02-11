/**
 * @file linear_svm_main.cpp
 * @author Ayush Chamoli
 *
 * Main executable for linear support vector machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_svm.hpp"

#include <ensmallen.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::util;

PROGRAM_INFO("Linear Support Vector Machine",
    // Short description
    "An implementation of linear support vector machine for multi-class "
    "classification. Given labeled data, a model can be trained and saved "
    "for future use; or, a pre-trained model can be used to classify new"
    "points.",
    // Long description
    "An implementation of L2-regularized linear support vector machine using"
    "either L-BFGS optimizer or Parallel SGD (stochastic gradient descent). "
    "This program is able to train a model, load an existing model, and give "
    "predictions (and optionally their accuracy) for test data."
    "\n\n"
     "This program allows loading a linear svm model (via the " +
     PRINT_PARAM_STRING("input_model") + " parameter) "
     "or training a linear svm model given training data (specified "
     "with the " + PRINT_PARAM_STRING("training") + " parameter), or both "
     "those things at once. In addition, this program allows classification on "
     "a test dataset (specified with the " + PRINT_PARAM_STRING("test") + " "
     "parameter) and the classification results may be saved with the " +
     PRINT_PARAM_STRING("predictions") + " output parameter."
     " The trained linear svm model may be saved using the " +
     PRINT_PARAM_STRING("output_model") + " output parameter."
     "\n\n"
     "The " + PRINT_PARAM_STRING("labels") + " parameter may be used to"
     " specify a matrix of labels."
     "\n\n"
     "When a model is being trained, there are many options."
     " The number of classes can be manually specified with the" +
     PRINT_PARAM_STRING("number_of_classes") + "parameter, L2 regularization "
     "(to prevent overfitting) can be specified with the " +
     PRINT_PARAM_STRING("lambda") + " option, and the "
     "optimizer used to train the model can be specified with the " +
     PRINT_PARAM_STRING("optimizer") + " parameter.  Available options are "
     "'psgd' (parallel stochastic gradient descent) and "
     "'lbfgs' (the L-BFGS optimizer).  "
     "There are also various parameters for the optimizer; the " +
     PRINT_PARAM_STRING("max_iterations") + " parameter specifies the maximum "
     "number of allowed iterations, and the " +
     PRINT_PARAM_STRING("tolerance") + " parameter specifies the tolerance for "
     "convergence.  For the SGD optimizer, the " +
     PRINT_PARAM_STRING("step_size") + " parameter controls the step size taken "
     "at each iteration by the optimizer.  The batch size for SGD is controlled "
     "with the " + PRINT_PARAM_STRING("batch_size") + " parameter. If the "
     "objective function for your data is oscillating between Inf and 0, the "
     "step size is probably too large.  There are more parameters for the "
     "optimizers, but the C++ interface must be used to access these."
     "\n\n"
     "For SGD, an iteration refers to a single point. So to take a single pass "
     "over the dataset with SGD, " + PRINT_PARAM_STRING("max_iterations") +
     " should be set to the number of points in the dataset."
     "\n\n"
     "Optionally, the model can be used to predict the responses for another "
     "matrix of data points, if " + PRINT_PARAM_STRING("test") + " is "
     "specified.  The " + PRINT_PARAM_STRING("test") + " parameter can be "
     "specified without the " + PRINT_PARAM_STRING("training") + " parameter, "
     "so long as an existing linear svm model is given with the " +
     PRINT_PARAM_STRING("input_model") + " parameter.  The output predictions "
     "from the linear svm model may be saved with the " +
     PRINT_PARAM_STRING("predictions") + " parameter." +
     "\n\n"
     "For example, to train a linear svm model on the data " +
     PRINT_DATASET("dataset") + " with labels " + PRINT_DATASET("labels") +
     " with a maximum of 1000 iterations for training, saving the trained model "
     "to " + PRINT_MODEL("sr_model") + ", the following command can be used: "
     "\n\n" +
     PRINT_CALL("linear_svm", "training", "dataset", "labels", "labels",
         "lambda", 0.1, "output_model", "lsvm_model") +
     "\n\n"
     "Then, to use " + PRINT_MODEL("sr_model") + " to classify the test points "
     "in " + PRINT_DATASET("test_points") + ", saving the output predictions to"
     " " + PRINT_DATASET("predictions") + ", the following command can be used:"
     "\n\n" +
     PRINT_CALL("linear_svm", "input_model", "lsvm_model", "test",
         "test_points", "predictions", "predictions"),
     SEE_ALSO("@softmax_regression", "#softmax_regression"),
     SEE_ALSO("@random_forest", "#random_forest"),
     SEE_ALSO("Support Vector Machine on Wikipedia",
         "https://en.wikipedia.org/wiki/Support-vector_machine"),
     SEE_ALSO("mlpack::svm::LinearSVM C++ class documentation",
         "@doxygen/classmlpack_1_1svm_1_1LinearSVM.html"));

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set (the matrix "
                            "of predictors, X).", "t");
PARAM_UROW_IN("labels", "A matrix containing labels (0 or 1) for the points "
                        "in the training set (y).", "l");

// Model loading/saving.
PARAM_MODEL_IN(LinearSVM<>, "input_model", "File containing existing "
    "model (parameters).", "m");
PARAM_MODEL_OUT(LinearSVM<>, "output_model", "File to save trained "
    "linear svm model to.", "M");

// Optimizer parameters.
PARAM_DOUBLE_IN("lambda", "L2-regularization parameter for training.", "L",
                0.0);
PARAM_STRING_IN("optimizer", "Optimizer to use for training ('lbfgs' or "
    "'sgd').", "O", "lbfgs");
PARAM_DOUBLE_IN("tolerance", "Convergence tolerance for optimizer.", "e",
    1e-10);
PARAM_INT_IN("max_iterations", "Maximum iterations for optimizer (0 indicates "
    "no limit).", "n", 10000);
PARAM_DOUBLE_IN("step_size", "Step size for SGD optimizer.",
    "s", 0.01);
PARAM_INT_IN("batch_size", "Batch size for SGD.", "b", 64);

