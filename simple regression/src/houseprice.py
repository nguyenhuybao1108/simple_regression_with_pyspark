from pyspark.sql.types import (
    StructField,
    StructType,
    IntegerType,
    DoubleType,
    StringType,
)
import pandas as pd
from regression import Regression

if __name__ == "__main__":
    a = {}
    filepath = (
        "/Users/baonguyen/Downloads/course/scalable/lab/lab11/datasets/houseprice.csv"
    )
    df = pd.read_csv(filepath)
    for column, dt in zip(df.dtypes.index, df.dtypes):
        if dt == "int64":
            a[column] = IntegerType()
        elif dt == "float64":
            a[column] = DoubleType()
        else:
            a[column] = StringType()
    schema = StructType([StructField(col, a[col], True) for col in df.columns])
    model = Regression(
        filepath=filepath,
        schema=schema,
        label="SalePrice",
    )
    model.eda()
    model.preprocessing(threshold=600)
    model.vector_assembler()
    model.cross_validation()
