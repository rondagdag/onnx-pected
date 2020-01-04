using Microsoft.ML.Data;
using System;

namespace TaxiFarePrediction.DataStructures
{
    public class OrderWithDetails
    {
        // //OrderID,ProductID,UnitPrice,Quantity,Discount
        //[LoadColumn(0)]
        //public string OrderID;

        [LoadColumn(1)]
        public string CustomerID;

        [LoadColumn(2)]
        public string EmployeeID;

        //[LoadColumn(3)]
        //public DateTime OrderDate;

        //[LoadColumn(4)]
        //public DateTime RequiredDate;

        [LoadColumn(5)]
        public string ShipVia;

        //[LoadColumn(6)]
        //public float Freight;

        [LoadColumn(7)]
        public string ProductID;

        [LoadColumn(8)]
        public float UnitPrice;

        [LoadColumn(9)]
        public float Quantity;

        [LoadColumn(10)]
        public float Discount;
    }
}