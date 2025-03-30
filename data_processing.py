import os


def process_data():
    # 导入必要的库
    import pandas as pd
    from pyspark.ml import Pipeline
    from pyspark.ml.connect.tuning import CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor
    from pyspark.ml.tuning import ParamGridBuilder
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    from pyspark.sql.functions import split, col, regexp_extract, regexp_replace
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
    from pyspark.ml.functions import vector_to_array
    try:

    # 创建 SparkSession 并配置
        spark = SparkSession.builder \
            .appName("Improved Spark App") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.sql.autoBroadcastJoinThreshold", "10m") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.3") \
            .getOrCreate()


        # 定义分析数据模式

        
        schema = StructType([
            StructField("price", IntegerType(), True),
            StructField("area", IntegerType(), True),
            StructField("source-id", StringType(), True),
            StructField("layout", StringType(), True),
            StructField("floor", StringType(), True),
            StructField("pos1", StringType(), True),
            StructField("pos2", StringType(), True),
            StructField("community", StringType(), True),
            StructField("subway", StringType(), True)
        ])

        data = spark.createDataFrame([], schema)

        # 读取和合并不同的 CSV 文件
        for i in range(1, 9):
            path = f'./bj_danke_{i}.csv' 
            try:
                df = spark.read.csv(path, header=True, inferSchema=True).repartition(8)
                data = data.union(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")  

        # 在大型转换操作后添加缓存
        data = data.cache()

        data = data.dropDuplicates()
        # 删除包含任何缺失值的行
        data = data.na.drop()
        # 提取楼层，总楼层
        data = data.withColumn("current_floor", split(col("floor"),"/").getItem(0).cast("integer"))\
            .withColumn("total_floors", regexp_replace(split(col("floor"), "/").getItem(1), "层", "").cast("integer"))
        # 提取户型信息
        data = data.withColumn("bedrooms", regexp_extract(col("layout"), r'(\d+)室', 1).cast("integer"))\
            .withColumn("bathrooms", regexp_extract(col("layout"), r'(\d+)卫', 1).cast("integer"))
        # 提取地铁距离
        data = data.withColumn("port_distance", regexp_extract(col("subway"),r'(\d+)米',1).cast("integer"))
        # 每平米房价
        data = data.withColumn("price", col("price").cast("float"))\
            .withColumn("area", col("area").cast("float"))\
            .withColumn("unit_price", col("price") / col("area"))
        

        # 编码 - 只对位置和小区进行索引编码
        indexer = StringIndexer(inputCols=["pos1", "community"], 
                              outputCols=["pos1_index", "community_index"])

        # 组合特征 - 使用索引编码后的特征
        assembler = VectorAssembler(
            inputCols=["area", "current_floor", "total_floors", 
                      "bedrooms", "bathrooms", "port_distance", 
                      "pos1_index", "community_index"], 
            outputCol="features", 
            handleInvalid="skip")
        
        # 创建简化的管道
        pipeline = Pipeline(stages=[indexer, assembler])
        model = pipeline.fit(data)
        data_transformed = model.transform(data)

        # 获取分析数据 - 移除了 OHE 相关的列
        analysis_data = data_transformed.select(
            'price', 'area', 'pos1', 'community', 'subway',
            'current_floor', 'total_floors', 'bedrooms', 'bathrooms',
            'port_distance', 'unit_price'
        ).toPandas()
        

        
        train_data, test_data = data_transformed.randomSplit([0.8, 0.2], seed=42)

        # 线性回归
        lr = LinearRegression(featuresCol='features', labelCol='price', regParam=0.1)
        lr_model = lr.fit(train_data)

        print("线性回归模型评估：")
        predictions = lr_model.transform(test_data)
        evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction', metricName='rmse')
        rmse = evaluator.evaluate(predictions)
        print(f"RMSE: {rmse}")

        # 保存预测结果和相关数据
        pred_results = predictions.select('price', 'prediction').toPandas()
        residuals = pred_results['prediction'] - pred_results['price']
        plot_data = pd.DataFrame({
            'actual_price': pred_results['price'],
            'predicted_price': pred_results['prediction'],
            'residuals': residuals
        })
        # 创建输出目录
        os.makedirs('./plot_data', exist_ok=True)
        # 保存数据供后续可视化使用
        plot_data.to_csv('./plot_data/prediction_results.csv', index=False)
        analysis_data.to_csv('./plot_data/analysis_data.csv', index=False)
        # 保存R²分数
        with open('./plot_data/r2_score.txt', 'w') as f:
            f.write(str(lr_model.summary.r2))

        # 关闭 Spark 会话
        spark.stop()
        
        return True
    except Exception as e:
        print(f"数据处理错误: {str(e)}")
        return False

if __name__ == "__main__":
    process_data() 