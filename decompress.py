import subprocess
import sys

#-------------- Regarding the installation of the dependencies -------------------
try:
    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed = [r.decode().split('==')[0] for r in reqs.split()]
except:
    print("Please install/update pip")
    exit()

required = ["pyspark", "numpy", "psutil"]
for pkg in required:
    if pkg not in installed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
print("\n Everything installed \n")
#---------------------------------------------------------------------------------

from pyspark import *
from pyspark.ml import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.sql import *


def check(x):
    if x == "NA":
        return 0
    return int(x)


# create spark session, local[*]: it runs locally with as many worker threads as possible
if __name__ == '__main__':

    conf = SparkConf().setMaster("local[*]").setAppName('prediction')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    RDD0 = sc.textFile(sys.argv[1])
    try:
        if len(RDD0.first().split(",")) != 29:
            print("INVALID FILE FORMAT, run the program again with a properly formatted file")
            exit()
    except:
        print("\nFILE NOT FOUND, run the program again with an existing file \n")
        print("Notice that the script must be called with this exact format:")
        print("\tpython script.py C:/folder/.../file.csv.bz2")
        exit()

    # we split in columns each row
    RDD1 = RDD0.map(lambda x: x.split(","))
    # we keep only those which were not canceled => canceled==0 (not cancelled))
    # it has been detected that the possible missing vars are distance x18 and target x14
    RDD2 = RDD1.filter(lambda x: x[21] == "0" and x[14] != "NA" and x[18] != "NA")

    # taxiOut is either NA in the whole file or an actual int, we will deal with that
    # key: origin(17) + destiny(18) + week day(4), (SUBTRACT 1 SINCE WE START AT 0)
    # value: [DepDelay(16) + taxiOut(21) if it exists, distance(19), arrDelay(15), 1]
    RDD3 = RDD2.map(lambda x: [x[16] + x[17] + x[3], [int(x[15]) + check(x[20]), int(x[18]), int(x[14]), 1]])

    # now we want one rdd per key adding the attributes: departure delay, distance, arrival delay, number of flights
    RDD4 = RDD3.reduceByKey(lambda x, y: [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]])

    # now we change the key to the origin the 1st 3 letters of the key, the value to the number of flights
    RDD5 = RDD4.map(lambda x: [x[0][0:3], x[1][3]])

    # we add all the values from the same airport into 1 rdd
    RDD6 = RDD5.reduceByKey(lambda x, y: x + y)
    # RDD6 key=origin, value= total number of flights from the airport

    # RDD7 = RDD4.map(lambda x: [x[0][0:3], [x[0][-1], x[1][0] / x[1][3], x[1][1] / x[1][3], x[1][2] / x[1][3]]])
    RDD7 = RDD4.map(lambda x: [x[0][0:3], [int(x[0][-1]) - 1, x[1][0] / x[1][3], x[1][1] / x[1][3], x[1][2] / x[1][3]]])
    # to the RDD7 with key=origin and value=index(dayWeek), avg(depDelay), avg(distance), avg(arrDelay)

    # we append the number of flights total from its origin airport
    RDD8 = RDD7.leftOuterJoin(RDD6)

    # now rdd=(airport, ([weekday, depDelay, distance, arrDelay], size))
    RDD = RDD8.map(lambda x: x[1][0][:3] + [x[1][1]] + [x[1][0][3]])
    # we obtain rdd=[weekday, depDelay, distance, size, arrDelay]

    # transform to dataframe for linear regression
    df = RDD.toDF(["weekday", "depDelay", "distance", "size", "target"])
    # indexer = StringIndexer(inputCol="weekday", outputCol="weekday_index")
    # encoder = OneHotEncoder(inputCol="weekday_index", outputCol="weekday_cat")
    encoder = OneHotEncoder(inputCol="weekday", outputCol="weekday_cat")
    assembler = VectorAssembler(inputCols=["weekday_cat", "depDelay", "distance", "size"], outputCol="features")

    # pipeline = Pipeline(stages=[indexer, encoder, assembler])
    pipeline = Pipeline(stages=[encoder, assembler])
    model = pipeline.fit(df)
    data = model.transform(df)
    data = data.select("features", "target")

    # Split the data into training and test model with 70% obs. going in training and 30% in testing
    train_dataset, test_dataset = data.randomSplit([0.7, 0.3])

    # Create the Multiple Linear Regression object having feature column as features and Label column as target
    MLR = LinearRegression(featuresCol="features", labelCol="target")

    # Train the model on the training using fit() method.
    model = MLR.fit(train_dataset)
    trainingSummary = model.summary
    print("Training r2: %f" % trainingSummary.r2)

    # Find out coefficient value
    coefficient \
        = model.coefficients
    print("The coefficients of the model are : %a" % coefficient)

    # Find out intercept Value
    intercept = model.intercept
    print("The Intercept of the model is : %f" % intercept)

    # Predict the delay on test Dataset using the evaluate method
    pred = model.evaluate(test_dataset)
    evaluation = RegressionEvaluator(labelCol="target", predictionCol="prediction")

    # r2 - coefficient of determination for test set
    r2 = evaluation.evaluate(pred.predictions, {evaluation.metricName: "r2"})
    print("Validation r2: %.5f" % r2)
