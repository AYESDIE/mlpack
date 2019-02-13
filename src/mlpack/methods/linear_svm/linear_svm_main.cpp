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
    " L-BFGS optimizer. This program is able to train a model, "
    "load an existing model, and give predictions (and optionally"
    " their accuracy) for test data."
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
     PRINT_PARAM_STRING("lambda") + " option. "
     "The optimizer being used is L-BFGS"
     "There are various parameters for the optimizer; the " +
     PRINT_PARAM_STRING("max_iterations") + " parameter specifies the maximum "
     "number of allowed iterations, and the " +
     PRINT_PARAM_STRING("tolerance") + " parameter specifies the tolerance for "
     "convergence. There are more parameters for the "
     "optimizers, but the C++ interface must be used to access these."
     "\n\n"
     "For L-BFGS, an iteration refers to a single point. So to take a "
     "single pass over the dataset with SGD, "
     + PRINT_PARAM_STRING("max_iterations") + " should be set to the "
     "number of points in the dataset."
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
PARAM_DOUBLE_IN("tolerance", "Convergence tolerance for optimizer.", "e",
    1e-10);
PARAM_INT_IN("max_iterations", "Maximum iterations for optimizer (0 indicates "
    "no limit).", "n", 10000);

// Testing.
PARAM_MATRIX_IN("test", "Matrix containing test dataset.", "T");
PARAM_UROW_OUT("predictions", "Matrix to save predictions for test dataset "
    "into.", "p");
PARAM_UROW_IN("test_labels", "Matrix containing test labels.", "L");

// Softmax configuration options.
PARAM_INT_IN("max_iterations", "Maximum number of iterations before "
    "termination.", "n", 400);

PARAM_INT_IN("number_of_classes", "Number of classes for classification; if "
    "unspecified (or 0), the number of classes found in the labels will be "
    "used.", "c", 0);

PARAM_STRING_IN("mat_type", "Type of matrix to be used for storing the "
    "predictors ('dense' or 'sparse').", "d", "dense")

// Build the linear svm model given the parameters.
template<typename Model>
Model* TrainSVM(const size_t maxIterations);

// Test the accuracy of the model.
template<typename Model>
void TestSVM(const size_t numClasses, const Model& model);

static void mlpackMain()
{
  // Collect command-line options.
  const double lambda = CLI::GetParam<double>("lambda");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");

  // One of training and input_model must be specified.
  RequireAtLeastOnePassed({ "training", "input_model" }, true);

  if (CLI::HasParam("training"))
  {
    RequireAtLeastOnePassed({ "labels" }, true, "if training data is specified,"
        " labels must also be specified");
  }

  RequireAtLeastOnePassed({ "output_model", "predictions"},
      false, "no output will be saved");

  ReportIgnoredParam({{ "training", false }}, "labels");
  ReportIgnoredParam({{ "training", false }}, "max_iterations");
  ReportIgnoredParam({{ "training", false }}, "number_of_classes");
  ReportIgnoredParam({{ "training", false }}, "lambda");
  // Max Iterations needs to be positive.
  RequireParamValue<int>("max_iterations", [](int x) { return x >= 0; },
      true, "max_iterations must be positive or zero");

  // Tolerance needs to be positive.
  RequireParamValue<double>("tolerance", [](double x) { return x >= 0.0; },
      true, "tolerance must be positive or zero");

  // Lambda must be positive.
  RequireParamValue<double>("lambda", [](double x) { return x >= 0.0; },
      true, "lambda must be positive or zero");

  // Number of classes must be positive.
  RequireParamValue<int>("number_of_classes", [](int x) { return x >= 0; },
      true, "number of classes must be greater than or "
      "equal to 0 (equal to 0 in case of unspecified.)");

  // Make sure we have an output file of some sort.
  RequireAtLeastOnePassed({ "output_model", "predictions" }, false, "no results"
      " will be saved");

  LinearSVM<>* svm = TrainSVM<LinearSVM<>>(maxIterations);

  TestSVM(svm->NumClasses(), *svm);

  CLI::GetParam<LinearSVM<>*>("output_model") = svm;
}

template<typename Model>
void TestSVM(const size_t numClasses, const Model& model)
{
  // If there is no test set, there is nothing to test on.
  if (!CLI::HasParam("test"))
  {
    ReportIgnoredParam({{ "test", false }}, "test_labels");
    ReportIgnoredParam({{ "test", false }}, "predictions");

    return;
  }

  // Get the test dataset, and get predictions.
  arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

  arma::Row<size_t> predictLabels;
  model.Classify(testData, predictLabels);

  // Save predictions, if desired.
  if (CLI::HasParam("predictions"))
    CLI::GetParam<arma::Row<size_t>>("predictions") = std::move(predictLabels);

  // Calculate accuracy, if desired.
  if (CLI::HasParam("test_labels"))
  {
    arma::Row<size_t> testLabels =
        std::move(CLI::GetParam<arma::Row<size_t>>("test_labels"));

    if (testData.n_cols != testLabels.n_elem)
    {
      Log::Fatal << "Test data given with " << PRINT_PARAM_STRING("test")
          << " has " << testData.n_cols << " points, but labels in "
          << PRINT_PARAM_STRING("test_labels") << " have " << testLabels.n_elem
          << " labels!" << endl;
    }

    vector<size_t> bingoLabels(numClasses, 0);
    vector<size_t> labelSize(numClasses, 0);
    for (arma::uword i = 0; i != predictLabels.n_elem; ++i)
    {
      if (predictLabels(i) == testLabels(i))
      {
        ++bingoLabels[testLabels(i)];
      }
      ++labelSize[testLabels(i)];
    }

    size_t totalBingo = 0;
    for (size_t i = 0; i != bingoLabels.size(); ++i)
    {
      Log::Info << "Accuracy for points with label " << i << " is "
          << (bingoLabels[i] / static_cast<double>(labelSize[i])) << " ("
          << bingoLabels[i] << " of " << labelSize[i] << ")." << endl;
      totalBingo += bingoLabels[i];
    }

    Log::Info << "Total accuracy for all points is "
        << (totalBingo) / static_cast<double>(predictLabels.n_elem) << " ("
        << totalBingo << " of " << predictLabels.n_elem << ")." << endl;
  }
}


template<typename Model>
Model* TrainSVM(const size_t maxIterations)
{
  using namespace mlpack;

  Model* svm;
  if (CLI::HasParam("input_model"))
  {
    svm = CLI::GetParam<Model*>("input_model");
  }
  else
  {
    arma::mat trainData = std::move(CLI::GetParam<arma::mat>("training"));
    arma::Row<size_t> trainLabels =
            std::move(CLI::GetParam<arma::Row<size_t>>("labels"));

    if (trainData.n_cols != trainLabels.n_elem)
      Log::Fatal << "Samples of input_data should same as the size of "
                 << "input_label." << endl;

    size_t numClasses = (size_t) CLI::GetParam<int>("number_of_classes");

    // Verify if the numClasses are set to default
    if (numClasses == 0)
    {
      const set<size_t> unique_labels(begin(trainLabels),
                                      end(trainLabels));
      numClasses = unique_labels.size();
    }

    const size_t numBasis = 5;
    ens::L_BFGS optimizer(numBasis, maxIterations);
    svm = new Model(trainData, trainLabels, numClasses,
        CLI::GetParam<double>("lambda"), std::move(optimizer));
  }

  return svm;
}

