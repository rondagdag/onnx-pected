using Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using TaxiFarePrediction.DataStructures;

namespace GenerateONNX_AutoML
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Create, train, evaluate and save a model
            BuildTrainEvaluateAndSaveModel(mlContext);

            // Make a single test prediction loading the model from .ZIP file
            TestSinglePrediction(mlContext);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static string BaseDatasetsRelativePath = @"Data";
        private static string DataRelativePath = $"{BaseDatasetsRelativePath}/order-details.csv";
        private static string DataPath = GetAbsolutePath(DataRelativePath);
        private static string LabelColumnName = "PayFullPrice";
        private static uint ExperimentTime = 10;

        private static readonly string MODEL_NAME = "discountPred.onnx";
        private static readonly string modelPath = Directory.GetCurrentDirectory() + @"\" + MODEL_NAME;
        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration

            string connectionString = @"Data Source=(LocalDB)\MSSQLLocalDB;Database=Northwind;Integrated Security=True;Connect Timeout=30";

            string sqlCommand = @"SELECT OrderID, 
                                    CAST([ProductID] as varchar) as ProductID,
                                    CAST([UnitPrice] as REAL) as UnitPrice,
                                    CAST([Quantity] as REAL) as Quantity,
                                    CAST([Discount] as varchar) as Discount 
                                    FROM [dbo].[Order Details]";

            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);

            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<OrderDetails>();

            IDataView dataview = loader.Load(dbSource);

            //IDataView dataview = mlContext.Data.LoadFromTextFile<OrderDetails>(DataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.Conversion.MapValue("PayFullPrice",
                new[] { new KeyValuePair<string, bool>("0", true) }, "Discount")
                .Append(mlContext.Transforms.DropColumns("Discount"));
            
            var transformedDataView = pipeline.Fit(dataview).Transform(dataview);

            using (var stream = File.Create("transformedData.tsv"))
            {
                mlContext.Data.SaveAsText(transformedDataView, stream);
            }

            ConsoleHelper.ShowDataViewInConsole(mlContext, transformedDataView);

            //// STEP 2: Initialize our user-defined progress handler that AutoML will 
            // invoke after each model it produces and evaluates.
            var progressHandler = new BinaryExperimentProgressHandler();

            //// STEP 3: Run AutoML regression experiment
            ConsoleHelper.ConsoleWriteHeader("=============== Training the model ===============");
            Console.WriteLine($"Running AutoML regression experiment for {ExperimentTime} seconds...");

            ExperimentResult<BinaryClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateBinaryClassificationExperiment(ExperimentTime)
                .Execute(transformedDataView, LabelColumnName, progressHandler: progressHandler);

            // Print top models found by AutoML
            Console.WriteLine();
            PrintTopModels(experimentResult);

            //// STEP 4: Evaluate the model and print metrics

            ConsoleHelper.ConsoleWriteHeader("===== Evaluating model's accuracy with test data =====");
            RunDetail<BinaryClassificationMetrics> best = experimentResult.BestRun;
            ITransformer trainedModel = best.Model;

            //// STEP 5: Save/persist the trained model - convonnx


            using (var stream = File.Create(MODEL_NAME))
            {
                mlContext.Model.ConvertToOnnx(trainedModel, dataview, stream);
            }
            Console.WriteLine("The model is saved to {0}", MODEL_NAME);

            return trainedModel;

        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            ConsoleHelper.ConsoleWriteHeader("=============== Testing prediction engine ===============");


            var session = new InferenceSession(modelPath);

            /*cont onnx*/

            var inputMeta = session.InputMetadata;

            var container = new List<NamedOnnxValue>();
            //10298,36,15.20,40,0.25
            container.Add(GetNamedOnnxValue<string>(inputMeta, "ProductID", "63"));
            container.Add(GetNamedOnnxValue<float>(inputMeta, "UnitPrice", 35.1f));
            container.Add(GetNamedOnnxValue<float>(inputMeta, "Quantity", 80f));
            container.Add(GetNamedOnnxValue<string>(inputMeta, "Discount", "0"));
            /* output onnx*/
            var result = session.Run(container);
            var output = result.First(x => x.Name == "PredictedLabel0").AsTensor<bool>().GetValue(0);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted Pay Full Price: {output:0.####}, actual : false");
            Console.WriteLine($"**********************************************************************");

            container = new List<NamedOnnxValue>();
            //10298,36,15.20,40,0.25
            container.Add(GetNamedOnnxValue<string>(inputMeta, "ProductID", "11"));
            container.Add(GetNamedOnnxValue<float>(inputMeta, "UnitPrice", 14f));
            container.Add(GetNamedOnnxValue<float>(inputMeta, "Quantity", 12f));
            container.Add(GetNamedOnnxValue<string>(inputMeta, "Discount", "0"));
            /* output onnx*/
            result = session.Run(container);
            output = result.First(x => x.Name == "PredictedLabel0").AsTensor<bool>().GetValue(0);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted Pay Full Price: {output:0.####}, actual : true");
            Console.WriteLine($"**********************************************************************");
        }

        private static NamedOnnxValue GetNamedOnnxValue<T>(IReadOnlyDictionary<string, NodeMetadata> inputMeta, string column, T value)
        {
            T[] inputDataInt = new T[] { value };
            var tensor = new DenseTensor<T>(inputDataInt, inputMeta[column].Dimensions);
            var namedOnnxValue = NamedOnnxValue.CreateFromTensor<T>(column, tensor);
            return namedOnnxValue;
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        /// <summary>
        /// Print top models from AutoML experiment.
        /// </summary>
        private static void PrintTopModels(ExperimentResult<RegressionMetrics> experimentResult)
        {
            // Get top few runs ranked by R-Squared.
            // R-Squared is a metric to maximize, so OrderByDescending() is correct.
            // For RMSE and other regression metrics, OrderByAscending() is correct.
            var topRuns = experimentResult.RunDetails
                .Where(r => r.ValidationMetrics != null && !double.IsNaN(r.ValidationMetrics.RSquared))
                .OrderByDescending(r => r.ValidationMetrics.RSquared).Take(3);

            Console.WriteLine("Top models ranked by R-Squared --");
            ConsoleHelper.PrintRegressionMetricsHeader();
            for (var i = 0; i < topRuns.Count(); i++)
            {
                var run = topRuns.ElementAt(i);
                ConsoleHelper.PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
            }
        }

        private static void PrintTopModels(ExperimentResult<BinaryClassificationMetrics> experimentResult)
        {
            // Get top few runs ranked by R-Squared.
            // R-Squared is a metric to maximize, so OrderByDescending() is correct.
            // For RMSE and other regression metrics, OrderByAscending() is correct.
            var topRuns = experimentResult.RunDetails
                .Where(r => r.ValidationMetrics != null && !double.IsNaN(r.ValidationMetrics.Accuracy))
                .OrderByDescending(r => r.ValidationMetrics.Accuracy).Take(3);

            Console.WriteLine("Top models ranked by Accuracy --");
            ConsoleHelper.PrintBinaryClassificationMetricsHeader();
            for (var i = 0; i < topRuns.Count(); i++)
            {
                var run = topRuns.ElementAt(i);
                ConsoleHelper.PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
            }
        }

    }
}
