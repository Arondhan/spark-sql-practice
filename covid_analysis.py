import csv
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("covid").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = (spark.read.csv("covid-data.csv", header=True, inferSchema=True)
      .withColumn("date", F.to_date("date"))
      .filter(~F.col("iso_code").startswith("OWID_"))) 


def save(res, name):
    res.show(truncate=False)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(res.columns)
        wr.writerows(res.collect())


t1 = (df.filter(F.col("date") == "2021-03-31")
      .select("iso_code",
              F.col("location").alias("страна"),
              F.round(F.col("total_cases") / F.col("population") * 100, 2).alias("процент_переболевших"))
      .orderBy(F.desc("процент_переболевших"))
      .limit(15))
save(t1, "task1")

week = df.filter(F.col("date").between("2021-03-25", "2021-03-31"))
w = Window.partitionBy("location").orderBy(F.desc("new_cases"))
t2 = (week.withColumn("rn", F.row_number().over(w))
      .filter("rn = 1")
      .select(F.col("date").alias("число"),
              F.col("location").alias("страна"),
              F.col("new_cases").cast("long").alias("кол_во_новых_случаев"))
      .orderBy(F.desc("кол_во_новых_случаев"))
      .limit(10))
save(t2, "task2")

wr = Window.orderBy("date")
t3 = (df.filter((F.col("iso_code") == "RUS") & F.col("date").between("2021-03-24", "2021-03-31"))
      .withColumn("вчера", F.lag("new_cases").over(wr))
      .select(F.col("date").alias("число"),
              F.col("вчера").cast("long").alias("кол_во_новых_случаев_вчера"),
              F.col("new_cases").cast("long").alias("кол_во_новых_случаев_сегодня"))
      .withColumn("дельта", F.col("кол_во_новых_случаев_сегодня") - F.col("кол_во_новых_случаев_вчера"))
      .filter(F.col("число") >= "2021-03-25")
      .orderBy("число"))
save(t3, "task3")

spark.stop()
