using Microsoft.ML.Data;

namespace TaxiFarePrediction.DataStructures
{
    public class OrderDetails
    {
        // //OrderID,ProductID,UnitPrice,Quantity,Discount

        [LoadColumn(1)]
        public string ProductID;

        [LoadColumn(2)]
        public float UnitPrice;

        [LoadColumn(3)]
        public float Quantity;

        [LoadColumn(4)]
        public string Discount;
    }
}