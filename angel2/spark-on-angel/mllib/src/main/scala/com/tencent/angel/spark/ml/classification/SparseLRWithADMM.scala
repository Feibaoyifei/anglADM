/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

package com.tencent.angel.spark.ml

import java.util.Date

import com.tencent.angel.spark.models.MLModel
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.mllib.optimization.{L1Updater, Optimizer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.linalg.{DenseVector, OneHotVector}
import com.tencent.angel.spark.ml.common.{Instance, Learner, LogisticGradient}
import com.tencent.angel.spark.ml.optimize.ADMM
import com.tencent.angel.spark.ml.util.LoadClassificationFile.{computeNumFeatures, parseLibSVMFile}
import com.tencent.angel.spark.ml.util._
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.storage.StorageLevel

/**
 * Sparse Logistic Regression is good at training sparse and high dimension Logistic Regression
 * Model with ADMM(Alternating Direction Method of Multipliers) optimizer.
 *
 * This algorithm only supports binary classification task right now. And the feature of train
 * data must be one hot feature(each dimension of feature must be either 0 or 1). We have train
 * 100 million dimension level model in 2~3 hours.
 *
 */
class SparseLRWithADMM extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  private var partitionNum: Int = _
  private var sampleRate: Double = _
  private var regParam = 0.0
  private var maxIter = 15
  private var numSubModels = 100
  private var rho = 0.01

  /**
   * set parallel number in spark
   */
  def setPartitionNum(num: Int): this.type = {
    partitionNum = num
    this
  }

  /**
   * set sample rate of data for train. Default 1.0
   */
  def setSampleRate(fraction: Double): this.type = {
    sampleRate = fraction
    this
  }

  /**
   * set the regularization parameter. Default 0.0
   */
  def setRegParam(reg: Double): this.type = {
    regParam = reg
    this
  }

  /**
   * set maximum number of iteration. Default 15
   */
  def setMaxIter(num: Int): this.type = {
    maxIter = num
    this
  }

  /**
   * Set the number of sub models to be trained in parallel. Default 20.
   */
  def setNumSubModels(num: Int): this.type = {
    numSubModels = num
    this
  }
  /**
   * Set rho which is the augmented Lagrangian parameter.
   * kappa = regParam / (rho * numSubModels), if the absolute value of element in model
   * is less than kappa, this element will be assigned to zero.
   * So kappa should be less than 0.01 or 0.001.
   */
  def setRho(factor: Double): this.type = {
    rho = factor
    this
  }

  def train(input: String, testSet: String): MLModel = {
    val tmpInstances = DataLoader.loadOneHotInstance(input, partitionNum, sampleRate, 1).rdd
        .map { row =>
          Tuple2(row.getString(0).toDouble, row.getAs[mutable.WrappedArray[Long]](1).toArray)
        }
    //val tmpInstances=DataLoader.loadLibsvm(input,partitionNum,sampleRate,-1);
    //val featLength = tmpInstances.map { case (label, feature) => feature.max }.max
   val featLength = 123
    println(s"feat length: $featLength")

    val instances = tmpInstances.map { case (label, feat) => (label, new OneHotVector(featLength, feat)) }

    val testRDD = if (testSet != null && testSet != "") {
      DataLoader.loadOneHotInstance(testSet, partitionNum, sampleRate, featLength - 1).rdd
        .map { row =>
          Tuple2(row.getString(0).toDouble, row.getAs[mutable.WrappedArray[Long]](1).toArray)
        }.map { case (label, feat) => (label, new OneHotVector(featLength, feat)) }
    } else {
      null
    }

    val lr = new ADMM(new LogisticGradient, new L1Updater)
      .setRegParam(regParam)
      .setNumIterations(maxIter)
      .setNumSubModels(numSubModels)
      .setRho(rho)
      .setTestSet(testRDD)

    val initModel = new DenseVector(new Array[Double](featLength.toInt))
    val (weight, lossHistory) = lr.optimize(instances, initModel)

    println(s"lr loss history: ${lossHistory.mkString(" ")}")
    val lrModel = new SparseModel(weight)
    lrModel
  }

  /*def predict(input: String, output: String, model: MLModel): Unit = {
    val featSize = model.asInstanceOf[SparseModel].weight.length - 1
    val instances = DataLoader.loadOneHotInstance(input, partitionNum, sampleRate, featSize - 1)
    println(s"predict instance count: ${instances.count()}")

    val predictDF = model.predict(instances)
    DataSaver.save(predictDF, output)
  }
*/
  override def optimizer: Optimizer = ???

  override protected def createModel(weights: linalg.Vector, intercept: Double): LogisticRegressionModel = ???
}

class SparseModel(val weight: DenseVector) extends MLModel {

  override def save(path: String): Unit = {
    val sc = SparkContext.getOrCreate()

    val modelPath = path + "/weight"
    val conf = sc.hadoopConfiguration
    val fs = new Path(modelPath).getFileSystem(conf)
    if (fs.exists(new Path(modelPath))) {
      fs.delete(new Path(modelPath), true)
    }

    sc.parallelize(weight.toArray, 10).zipWithIndex()
      .map { case (w, index) => index + ":" + w}
      .saveAsTextFile(modelPath)
  }

  override def predict(input: DataFrame): DataFrame = {
    val sc = SparkContext.getOrCreate()
    val modelBC = sc.broadcast(weight)
    val modelLength = weight.length
    val predictUDF = udf { (feature: mutable.WrappedArray[Int]) =>
      val weightSum = feature.filter(index => index < modelLength)
        .map { index => modelBC.value.values(index) }.sum
      1.0 / (1 + math.exp(-1 * weightSum))
    }
    input.withColumn(DFStruct.PROB, predictUDF(col(DFStruct.FEATURE))).drop(DFStruct.FEATURE)
  }
  override def evaluate(testSet: DataFrame, evaluator: Evaluator): Double = 0.5

  override def summary(): String = null
}


object SparseLRWithADMM {

  def main(args: Array[String]): Unit = {
    println(s"Start Sparse Logistic Regression Processing...")
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse(ParamKeys.INPUT, "hdfs://node0:9820/user/data/rcv1train")
    val sampleRate = params.getOrElse(ParamKeys.SAMPLE_RATE, "1.0").toDouble
    val partitionNum = params.getOrElse(ParamKeys.PARTITION_NUM, "20").toInt
    val numSubModel = params.getOrElse("numSubModel", "1").toInt
    val regParam = params.getOrElse(ParamKeys.REG_PARAM, "0.0").toDouble
    val rho = params.getOrElse("rho", "0.1").toDouble
    val maxIter = params.getOrElse(ParamKeys.MAX_ITER, "1").toInt
    val testSet = params.getOrElse(ParamKeys.TEST_SET, "hdfs://node0:9820/user/data/rcv1test.txt")
    val output = params.getOrElse(ParamKeys.OUTPUT, "hdfs://node0:9820/user/lzz/out")
    val psConf = new SparkConf()
     .set("spark.ps.mode", "LOCAL").set("spark.ps.tmp.path", "file:///tmp/stage")
      .set("spark.ps.out.path", "file:///tmp/output").set("spark.ps.model.path", "file:///tmp/model")
    val conf = psConf.setAppName(this.getClass.getSimpleName).setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    PSContext.getOrCreate(spark.sparkContext)
    val lr = new SparseLRWithADMM()
        .setPartitionNum(partitionNum).setSampleRate(sampleRate)
        .setNumSubModels(numSubModel)
        .setRegParam(regParam)
        .setRho(rho)
        .setMaxIter(maxIter)
    val start_time = new Date().getTime
    val mo=lr.train(input,testSet)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
   spark.stop()
  }
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
}
