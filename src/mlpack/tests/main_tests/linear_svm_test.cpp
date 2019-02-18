/**
 * @file linear_svm_test.cpp
 * @author Ayush Chamoli
 *
 * Test mlpackMain() of linear_svm_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "LinearSVM";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/linear_svm/linear_svm_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LinearSVMTestFixture
{
public:
    LinearSVMTestFixture()
    {
      // Cache in the options for this program.
      CLI::RestoreSettings(testName);
    }

    ~LinearSVMTestFixture()
    {
      // Clear the settings.
      bindings::tests::CleanMemory();
      CLI::ClearSettings();
    }
};

BOOST_FIXTURE_TEST_SUITE(LinearSVMMainTest,
                         LinearSVMTestFixture);


/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
BOOST_AUTO_TEST_CASE(LinearSVMOutputDimensionTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
      testSize);

  // Check prediction have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
      1);
}


/**
 * Ensure that labels are necessarily passed when training.
 */
BOOST_AUTO_TEST_CASE(LinearSVMLabelsLessDimensionTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(LinearSVMModelReuseTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> predictions;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
      CLI::GetParam<LinearSVM<>*>("output_model"));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
      testSize);

  // Check predictions have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
      1);

  // Check that initial predictions and final predicitons matrix
  // using saved model are same.
  CheckMatrices(predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
}

/**
 * Ensure that max_iterations is always non-negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMMaxItrTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_iterations", (int) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that lambda is always non-negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMLambdaTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("lambda", (double) -0.1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that number_of_classes is always positive.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNumClassesTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("number_of_classes", (int) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that output object parameters are
 * different for different lambda values.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffLambdaTest)
{
  // Train SR for lambda 0.1.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("lambda", (double) 0.1);

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = CLI::GetParam<LinearSVM<>*>("output_model")->Parameters();

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Train SR for lamda 0.9.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("lambda", (double) 0.9);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial parameters and final parameters matrix
  // using saved model are different.
  for (size_t i = 0; i < modelParam.n_elem; ++i)
  {
    BOOST_REQUIRE_NE(modelParam[i],
        CLI::GetParam<LinearSVM<>*>("output_model")->Parameters()[i]);
  }
}

/**
 * Check that output object parameters are different for different numbers of
 * max_iterations.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffMaxItrTest)
{
  // Train SR for lambda 0.1.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("max_iterations", (int) 500);

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = CLI::GetParam<LinearSVM<>*>("output_model")->Parameters();

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Train SR for lamda 0.9.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_iterations", (int) 1000);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial parameters and final parameters matrix
  // using saved model are different.
  for (size_t i = 0; i < modelParam.n_elem; ++i)
  {
    BOOST_REQUIRE_NE(modelParam[i],
        CLI::GetParam<LinearSVM<>*>("output_model")->Parameters()[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
