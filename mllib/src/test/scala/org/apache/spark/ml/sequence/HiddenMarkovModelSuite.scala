/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.sequence

import java.util.Random

import breeze.linalg.{DenseVector => BDV}
import breeze.stats.distributions.{Multinomial => BrzMultinomial}

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.sequence.HiddenMarkovModelSuite._
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql._

class HiddenMarkovModelSuite extends SparkFunSuite
  with MLlibTestSparkContext with DefaultReadWriteTest{

  @transient var dataset: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val pi = Vectors.dense(Array(0.5, 0.5))
    val A = new DenseMatrix(2, 2, Array(0.7, 0.3, 0.3, 0.7))
    val B = Matrices.dense(2, 2, Array(0.2, 0.8, 0.9, 0.1))

    dataset = spark.createDataFrame(generateMultinomialHMMInputWithLabel(pi, A, B, 100, 3, 9, 42))
  }

  def validatePrediction(predictionAndLabel: DataFrame): Unit = {
    val numOfErrorPredictions = predictionAndLabel.collect().foldLeft(0.0) { (r, c) =>
      c match {
        case Row(prediction: Vector, label: Vector) =>
          val diff = label.copy
          BLAS.axpy(-1.0, prediction, diff)
          diff.toArray.filterNot(_ == 0.0).size / label.size + r
      }
    }
    assert(numOfErrorPredictions < predictionAndLabel.count() / 10)
  }

  def validateModelFit(
      transitionData: DenseMatrix,
      emissionData: Matrix,
      model: HMMModel): Unit = {
    assert(model.transitionProb ~== transitionData absTol 0.05, "A mismatch")
    if(model.emissionType === "multinomial") {
      assert(model.asInstanceOf[MultinomialHMMModel].emissionProb ~== emissionData absTol 0.05, "B mismatch")
    }
  }

  def validateProbabilities(
      featureAndProbabilities: DataFrame,
      model: HMMModel): Unit = {
    featureAndProbabilities.collect().foreach {
      case Row(features: Matrix, probability: Double) =>
        val emissions = model.precomputeEmissions(features)
        val backward = model.backward(emissions.tail)
        val expected = (model.initialProb.toArray, emissions.head, backward.head).zipped
          .map((a, e, b) => a * e * b).sum
        assert((probability - expected).abs < 1.0e-4)
    }
  }

  test("params") {
    ParamsSuite.checkParams(new HiddenMarkovModel)
    val model = new MultinomialHMMModel("hmm",
      initialProb = Vectors.dense(Array(0.5, 0.5)),
      transitionProb = new DenseMatrix(2, 2, Array(0.7, 0.3, 0.3, 0.7)),
      emissionProb = Matrices.dense(2, 2, Array(0.2, 0.8, 0.9, 0.1)))
    ParamsSuite.checkParams(model)
  }

  test("hmm: default params") {
    val hmm = new HiddenMarkovModel
    assert(hmm.getLabelCol === "label")
    assert(hmm.getFeaturesCol === "features")
    assert(hmm.getPredictionCol === "prediction")
    assert(hmm.getProbabilityCol === "probability")
    assert(hmm.getSmoothing === 1.0)
    assert(hmm.getThreshold === 1.0)
  }

  test("HMM with Multinomial Emission") {
    val nSequences = 2000
    val pi = Vectors.dense(Array(0.5, 0.5))
    val A = new DenseMatrix(2, 2, Array(0.7, 0.3, 0.3, 0.7))
    val B = Matrices.dense(2, 2, Array(0.2, 0.8, 0.9, 0.1))

    val testDataset = spark.createDataFrame(generateMultinomialHMMInputWithLabel(
      pi, A, B, nSequences, 3, 9, 42))

    val hmm = new HiddenMarkovModel().setEmissionType("multinomial")
    hmm.setInitialModel(new MultinomialHMMModel(Identifiable.randomUID("hmm"),
      Vectors.dense(Array(0.5, 0.5)),
      new DenseMatrix(2, 2, Array(0.6, 0.4, 0.2, 0.8)),
      Matrices.dense(2, 2, Array(0.4, 0.6, 0.7, 0.3))).setEmissionType("multinomial"))
    val model = hmm.train(testDataset)

    validateModelFit(A, B, model)

    val validationDataset = spark.createDataFrame(generateMultinomialHMMInputWithLabel(
      pi, A, B, nSequences, 3, 9, 17))

    val predictionAndLabels = model.transform(validationDataset).select("prediction", "label")
    validatePrediction(predictionAndLabels)

    val featureAndProbabilities = model.transform(validationDataset)
      .select("features", "probability")
    validateProbabilities(featureAndProbabilities, model)
  }

  test("read/write") {
    def checkModelData(model: HMMModel, model2: HMMModel): Unit = {
      assert(model.transitionProb === model2.transitionProb)
    }
    val hmm = new HiddenMarkovModel()
    testEstimatorAndModelReadWrite(hmm, dataset, HiddenMarkovModelSuite.allParamSettings,
      checkModelData)
  }
}

object HiddenMarkovModelSuite {

  val allParamSettings: Map[String, Any] = Map(
    "predictionCol" -> "myPrediction"
  )

  private def calcLabel(p: Double, pi: Array[Double]): Int = {
    var sum = 0.0
    for (j <- 0 until pi.length) {
      sum += pi(j)
      if(p < sum) return j
    }
    -1
  }

  def generateMultinomialHMMInputWithLabel(
      pi: Vector,
      A: DenseMatrix,
      B: Matrix,
      nSequences: Int,
      minLength: Int,
      maxLength: Int,
      seed: Int,
      sample: Int = 1): Seq[LabeledSequence] = {

    val D = B.numRows
    val C = A.numRows
    val transition = A.toArray
    val emission = B.toArray.grouped(D).toArray
    val rnd = new Random(seed)
    for (i <- 0 until nSequences) yield {
      val length = minLength + rnd.nextInt(maxLength - minLength)
      val y = Array.ofDim[Int](length)
      y(0) = calcLabel(rnd.nextDouble(), pi.toArray)
      var prevY = y(0)
      for (j <- 1 until length) {
        y(j) = calcLabel(rnd.nextDouble(), transition.slice(prevY * C, (1 + prevY) * C))
        prevY = y(j)
      }

      val x = Array.ofDim[Double](length)
      for (j <- 0 until length) {
        val mult = BrzMultinomial(BDV(emission(y(j))))
        val emptyMap = (0 until D).map(x => (x, 0.0)).toMap
        val counts = emptyMap ++ mult.sample(sample).groupBy(x => x).map {
          case (index, reps) => (index, reps.size.toDouble)
        }
        x(j) = counts.toArray.sortWith((x, y) => x._2 > y._2).head._1
      }
      LabeledSequence(Vectors.dense(y.map(_.toDouble)), Matrices.dense(1, length, x))
    }
  }
}
