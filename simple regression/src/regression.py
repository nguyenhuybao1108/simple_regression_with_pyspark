from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField,
    StructType,
    IntegerType,
    DoubleType,
    StringType,
)
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when, count, col, isnan, isnull, mean, stddev
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    OneHotEncoder,
    StringIndexer,
    ChiSqSelector,
)
from pyspark.ml import regression
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd


class Regression:

    def __init__(self, filepath, schema, label):
        self.spark = SparkSession.builder.appName("HousePriceRegression").getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.df = self.spark.read.csv(filepath, header=True, schema=schema)
        self.df = self.df.drop("Id")

        self.label = label

        # Identify numerical and categorical columns
        self.num_col = [
            col_name
            for col_name, dt in self.df.dtypes
            if col_name != self.label and dt in ["int", "double"]
        ]
        self.cat_col = [
            col_name
            for col_name, dt in self.df.dtypes
            if col_name != self.label and dt == "string"
        ]

    def show(self):
        self.df.show(5, truncate=False)

    def eda(self):
        self.df.select(self.df.columns).summary().show(truncate=False)
        self.df = self.df.replace(["NA", "NULL", "NaN", ""], None)
        self.df.select(
            [
                count(
                    when(
                        col(c).isNull(),
                        c,
                    )
                ).alias(c)
                for c in self.df.columns
            ]
        ).show()

    def drop_columns_with_missing_values(self, threshold: int) -> None:
        missing_counts = (
            self.df.select(
                [count(when(col(c).isNull(), c)).alias(c) for c in self.df.columns]
            )
            .collect()[0]
            .asDict()
        )

        columns_to_drop = [
            col for col, count in missing_counts.items() if count > threshold
        ]
        self.df = self.df.drop(*columns_to_drop)

        # Update num_col and cat_col after dropping columns
        self.num_col = [
            col_name
            for col_name, dt in self.df.dtypes
            if col_name != self.label and dt in ["int", "double"]
        ]

        self.cat_col = [
            col_name
            for col_name, dt in self.df.dtypes
            if col_name != self.label and dt == "string"
        ]

    def string_indexer(self):
        indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_index")
            for col in self.cat_col
        ]
        return indexers

    def fill_missing(self, threshold: int = None):
        if threshold is not None:
            self.drop_columns_with_missing_values(threshold=threshold)

        if self.num_col:
            num_imputer = Imputer(
                inputCols=self.num_col, outputCols=self.num_col, strategy="median"
            )
            self.df = num_imputer.fit(self.df).transform(self.df)
        for col_name in self.cat_col:
            # Calculate mode by counting occurrences of each value
            mode_value_row = (
                self.df.groupBy(col_name).count().orderBy(F.desc("count")).collect()
            )

            if len(mode_value_row) > 0:
                # Find the first non-null mode value (excluding None)
                mode_value = None
                for row in mode_value_row:
                    if row[col_name] is not None:
                        mode_value = row[col_name]
                        break

                # If all values were None, mode_value will still be None here
                if mode_value is not None:
                    self.df = self.df.withColumn(
                        col_name,
                        when(col(col_name).isNull(), mode_value).otherwise(
                            col(col_name)
                        ),
                    )

    def handel_outlier(self):
        for col_name in self.num_col:
            # Calculate quartiles
            quantiles = self.df.approxQuantile(col_name, [0.25, 0.75], 0.05)
            q1 = quantiles[0]
            q3 = quantiles[1]

            # Calculate IQR
            iqr = q3 - q1

            # Calculate lower and upper bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Replace outliers with upper or lower bound
            self.df = self.df.withColumn(
                col_name,
                F.when(col(col_name) < lower_bound, lower_bound)
                .when(col(col_name) > upper_bound, upper_bound)
                .otherwise(col(col_name)),
            )

    def scale(self):
        for col_name in self.num_col:
            # Calculate mean and standard deviation
            mean_val = self.df.agg(mean(col(col_name))).collect()[0][0]
            stddev_val = self.df.agg(stddev(col(col_name))).collect()[0][0]

            # Apply Z-score normalization
            self.df = self.df.withColumn(
                col_name + "_scaled", (col(col_name) - mean_val) / stddev_val
            )
        self.scale_col = [
            col.name for col in self.df.schema.fields if col.name.endswith("_scaled")
        ]

    def onehot_encoder(self):
        self.string_indexer()
        encoders = [
            OneHotEncoder(inputCol=col + "_index", outputCol=col + "_onehot")
            for col in self.cat_col
        ]
        self.encoder_col = [
            col.name for col in self.df.schema.fields if col.name.endswith("_onehot")
        ]
        return encoders

    def vector_assembler(self):
        all_stages = self.string_indexer() + self.onehot_encoder()
        pipeline = Pipeline(stages=all_stages)
        self.df = pipeline.fit(self.df).transform(self.df)

        # Drop columns with null values before VectorAssembler
        columns_to_drop = []
        for col_name in self.scale_col + self.encoder_col:
            if self.df.filter(col(col_name).isNull()).count() > 0:
                columns_to_drop.append(col_name)

        self.df = self.df.drop(*columns_to_drop)
        ####UPDATE SCALE AND ENCODER COLUMN
        self.scale_col = [
            col.name for col in self.df.schema.fields if col.name.endswith("_scaled")
        ]
        self.encoder_col = [
            col.name for col in self.df.schema.fields if col.name.endswith("_onehot")
        ]
        idx_col = [
            col.name for col in self.df.schema.fields if col.name.endswith("_index")
        ]
        # Assemble remaining columns into 'features'
        self.df = (
            VectorAssembler(
                inputCols=self.scale_col + self.encoder_col,
                outputCol="features",
            )
            .transform(self.df)
            .drop(
                *self.num_col
                + idx_col
                + self.scale_col
                + self.cat_col
                + self.encoder_col
            )
        )

    def preprocessing(self, threshold):
        self.fill_missing(threshold=threshold)
        self.handel_outlier()
        self.scale()
        pass

    def train_test_split(self, test_size: float):
        training, test = self.df.randomSplit([1 - test_size, test_size], seed=123)
        return training, test

    def fit_transform(self):

        model = LinearRegression(featuresCol="features", labelCol=self.label)

        return model

    def cross_validation(self):
        ##=====build cross valiation model======
        lr = self.fit_transform()
        # parameter grid
        from pyspark.ml.tuning import ParamGridBuilder

        param_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, [0, 0.1, 0.5, 1])
            .addGrid(lr.elasticNetParam, [0, 0.1, 0.5, 1])
            .build()
        )

        # evaluator
        evaluator = RegressionEvaluator(
            predictionCol="prediction", labelCol=self.label, metricName="rmse"
        )
        training, test = self.train_test_split(test_size=0.2)
        # cross-validation model
        from pyspark.ml.tuning import CrossValidator

        cv = CrossValidator(
            estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5
        )
        cv_model = cv.fit(training)
        pred_training_cv = cv_model.transform(training)
        pred_test_cv = cv_model.transform(test)
        # performance on training data
        rmse_training = evaluator.evaluate(pred_training_cv)
        rmse_test = evaluator.evaluate(pred_test_cv)
        print(f"{rmse_training=}")
        print(f"{rmse_test=}")
        print(
            "Intercept: ",
            cv_model.bestModel.intercept,
            "\n",
            "coefficients: ",
            cv_model.bestModel.coefficients,
        )
        print(
            "best regParam: "
            + str(cv_model.bestModel._java_obj.getRegParam())
            + "\n"
            + "best ElasticNetParam:"
            + str(cv_model.bestModel._java_obj.getElasticNetParam())
        )
        print("pred_training_cv")
        pred_training_cv.select("SalePrice", "prediction", "features").show(
            5, truncate=False
        )
        print("pred_test_cv")
        pred_test_cv.select("SalePrice", "prediction", "features").show(
            5, truncate=False
        )
