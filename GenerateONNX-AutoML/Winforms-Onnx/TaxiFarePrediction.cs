﻿using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Math;
namespace WinForms_WinML_ONNX
{
    public partial class TaxiFarePrediction : Form
    {
        public TaxiFarePrediction()
        {
            InitializeComponent();
        }

        private void Predict()
        {

            var inputMeta = _session.InputMetadata;
            var container = new List<NamedOnnxValue>();

            container.Add(GetOnnxValue<float>(inputMeta, "PassengerCount", float.Parse(passengerCountTB.Text)));
            container.Add(GetOnnxValue<float>(inputMeta, "TripTime", float.Parse(tripDistanceTB.Text)));
            container.Add(GetOnnxValue<float>(inputMeta, "TripDistance", float.Parse(tripDistanceTB.Text)));
            container.Add(GetOnnxValue<float>(inputMeta, "FareAmount", 0f));

            var result = _session.Run(container);

            var output = result.First(x => x.Name == "Score0").AsTensor<float>().ToArray();
            var scores = result.Select(x => x.AsTensor<float>()).ToArray();
            var pred = output.Max();
            ShowResult(pred, output, 0);
        }


        private InferenceSession _session;
        private void LoadModel(string file)
        {
            _session = new InferenceSession(file);
            textUrl.Text = "LOADED!";
        }

        private string Stringify(float[] data)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < data.Length; i++)
            {
                if (i == 0) sb.Append("{\r\n\t");
                else if (i % 28 == 0)
                    sb.Append("\r\n\t");
                sb.Append($"{data[i],3:##0}, ");

            }
            sb.Append("\r\n}\r\n");
            return sb.ToString();
        }

        private void ShowResult(float prediction, float[] scores, double time, Func<double, double> conversion = null)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Scores:");

            for (int i = 0; i < scores.Length; i++)
            {
                double v = conversion == null ? scores[i] : conversion(scores[i]);
                sb.AppendLine($"\t{i}: {v}");
            }

            sb.AppendLine($"Prediction: {prediction}");
            // sb.AppendLine($"Time: {time}");
            labelPrediction.Text = prediction.ToString();

            textResponse.Text = "";
            textResponse.Text = sb.ToString();
        }

        private void Clear()
        {
            textResponse.Clear();
            labelPrediction.Text = "";
        }

        private static NamedOnnxValue GetOnnxValue<T>(IReadOnlyDictionary<string, NodeMetadata> inputMeta, string column, T value)
        {
            T[] inputData = new T[] { value };
            var tensor = new DenseTensor<T>(inputData, inputMeta[column].Dimensions);
            var namedOnnxValue = NamedOnnxValue.CreateFromTensor<T>(column, tensor);
            return namedOnnxValue;
        }

        private void ButtonLoad_Click(object sender, EventArgs e)
        {
            if (openFile.ShowDialog() == DialogResult.OK && File.Exists(openFile.FileName))
                LoadModel(openFile.FileName);
        }

        private void ButtonClear_Click(object sender, EventArgs e)
        {
            Clear();
        }

        private void ButtonRecognize_Click(object sender, EventArgs e)
        {
            Predict();
        }

        private void TextResponse_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
