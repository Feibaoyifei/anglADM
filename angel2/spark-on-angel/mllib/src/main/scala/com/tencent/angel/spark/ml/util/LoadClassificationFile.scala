package com.tencent.angel.spark.ml.util

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.DataSource
import org.apache.spark.sql.execution.datasources.text.TextFileFormat
import org.apache.spark.sql.functions.{length, not, trim}
import org.apache.spark.storage.StorageLevel

// 因为通用的线性分类模型支持0，1标签
object LoadClassificationFile {
  def loadLibSVMFile(
                      sc: SparkContext,
                      path: String,
                      numFeatures:Int,
                      minPartitions: Int): RDD[LabeledPoint] = {
    val parsed = parseLibSVMFile(sc, path, minPartitions)

    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      computeNumFeatures(parsed)
    }

    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }
  }

   def computeNumFeatures(rdd: RDD[(Double, Array[Int], Array[Double])]): Int = {
    rdd.map { case (label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1
  }

   def parseLibSVMFile(
                                      sc: SparkContext,
                                      path: String,
                                      minPartitions: Int): RDD[(Double, Array[Int], Array[Double])] = {
    sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map(parseLibSVMRecord)
  }

   def parseLibSVMFile(
                                      sparkSession: SparkSession, paths: Seq[String]): RDD[(Double, Array[Int], Array[Double])] = {
    val lines = sparkSession.baseRelationToDataFrame(
      DataSource.apply(
        sparkSession,
        paths = paths,
        className = classOf[TextFileFormat].getName
      ).resolveRelation(checkFilesExist = false))
      .select("value")

    import lines.sqlContext.implicits._

    lines.select(trim($"value").as("line"))
      .filter(not((length($"line") === 0).or($"line".startsWith("#"))))
      .as[String]
      .rdd
      .map(LoadClassificationFile.parseLibSVMRecord)
  }

  //将数据原先为 -1的标签转变为0 1的标签不变 提供给MLlib实现的LR使用
   def parseLibSVMRecord(line: String): (Double, Array[Int], Array[Double]) = {
    val items = line.split(' ')
     var label:Double = 0
     //TODO
     if (items.head.toDouble == -1.0)
       label= 0
     else
        label = items.head.toDouble

    val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
      val indexAndValue = item.split(':')
      val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
    val value = indexAndValue(1).toDouble
      (index, value)
    }.unzip

    // check if indices are one-based and in ascending order
    var previous = -1
    var i = 0
    val indicesLength = indices.length
    while (i < indicesLength) {
      val current = indices(i)
      require(current > previous, s"indices should be one-based and in ascending order;"
        + s""" found current=$current, previous=$previous; line="$line"""")
      previous = current
      i += 1
    }
    (label, indices.toArray, values.toArray)
  }

  /**
    * Loads labeled data in the LIBSVM format into an RDD[LabeledPoint], with the default number of
    * partitions.
    */

  /**
    * Loads binary labeled data in the LIBSVM format into an RDD[LabeledPoint], with number of
    * features determined automatically and the default number of partitions.
    */
  def loadLibSVMFile(sc: SparkContext, path: String): RDD[LabeledPoint] =
    loadLibSVMFile(sc, path, -1,4)


}
