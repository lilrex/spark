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

import scala.collection.mutable

import org.apache.hadoop.fs.Path

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkException
import org.apache.spark.sql.Dataset

private[sequence] trait HiddenMarkovModelParams extends MarginalTaggerParams
  with HasMaxIter with HasTol with HasStandardization with HasThreshold {

  final val smoothing: DoubleParam = new DoubleParam(this, "smoothing", "The smoothing parameter.",
    ParamValidators.gtEq(0))

  /** @group getParam */
  final def getSmoothing: Double = $(smoothing)

  final val emissionType: Param[String] = new Param[String](this, "emission", "The emission type" +
    " which is a string (case-sensitive). Supported options: multinomial (default) and gaussian.",
    ParamValidators.inArray[String](Set("multinomial", "gaussian").toArray))

  final def getEmissionType: String = $(emissionType)
}

class HiddenMarkovModel (override val uid: String)
  extends MarginalTagger[Matrix, HiddenMarkovModel, HMMModel]
  with HiddenMarkovModelParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("hmm"))

  def setSmoothing(value: Double): this.type = set(smoothing, value)
  setDefault(smoothing -> 1.0)

  def setEmissionType(value: String): this.type = set(emissionType, value)
  setDefault(emissionType -> "multinomial")

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  override def setThreshold(value: Double): this.type = set(threshold, value)
  setDefault(threshold -> 1.0)

  private var initialModel: Option[HMMModel] = None

  def setInitialModel(model: HMMModel): this.type = {
    initialModel = Some(model)
    this
  }

  /**
   * Train a model using the given dataset and parameters.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset Training dataset
   * @return Fitted model
   */
  override def train(dataset: Dataset[_]): HMMModel = {
    val supervisedHMM = trainSupervised(dataset)
    if(supervisedHMM.isDefined) {
      setInitialModel(supervisedHMM.get)
    }
    trainUnsupervised(dataset)
  }

  def trainSupervised(dataset: Dataset[_]): Option[HMMModel] = {
    val uid = Identifiable.randomUID("hmm")
    val sequences: RDD[LabeledSequence] = extractLabeledSequences(dataset).filter(_.label != null)

    val labelSummarizer = {
      val seqOp = (c: MultiClassSummarizer, instance: LabeledSequence) =>
        c.add(instance.label)
      val combOp = (c1: MultiClassSummarizer, c2: MultiClassSummarizer) =>
        c1.merge(c2)
      sequences.treeAggregate(new MultiClassSummarizer(${smoothing}))(seqOp, combOp)
    }

    val numClasses = labelSummarizer.numClasses
    if(numClasses > 0) {
      val initialProb = labelSummarizer.initialProb
      val transitionProb = labelSummarizer.transitionProb

      if (initialModel.isDefined && initialModel.get.numClasses != numClasses) {
        logWarning(s"Initial model parameters will be ignored!! As its number of classes" +
          s"${initialModel.get.numClasses} did not match the expected size ${numClasses}")
      }

      val model = ${emissionType} match {
        case "multinomial" =>
          val emissionSummarizer = {
            val seqOp = (c: MultinomialSummarizer, instance: LabeledSequence) =>
              c.add(instance)
            val combOp = (c1: MultinomialSummarizer, c2: MultinomialSummarizer) =>
              c1.merge(c2)
            sequences.treeAggregate(
              new MultinomialSummarizer(numClasses, ${smoothing}))(seqOp, combOp)
          }
          val numFeatures = emissionSummarizer.numFeatures
          val emissionProb = emissionSummarizer.emissionProb
          if (initialModel.isDefined && initialModel.get.numFeatures != numFeatures) {
            logWarning(s"Initial model parameters will be ignored!! As its number of features" +
              s"${initialModel.get.numFeatures} did not match the expected size ${numFeatures}")
          }
          new MultinomialHMMModel(uid, initialProb, transitionProb, emissionProb)
            .setEmissionType("multinomial")
        //      case "gaussian" =>
        case _ =>
          throw new UnknownError(s"Invalid emissionType: ${$(emissionType)}.")
      }
      Option(model)
    } else {
      None
    }
  }

  def trainUnsupervised(dataset: Dataset[_]): HMMModel = {
    val instances: RDD[LabeledSequence] = extractLabeledSequences(dataset)
    val uid = Identifiable.randomUID("hmm")

    val (initialProb, transitionProb) = initialModel match {
      case Some(model) =>
        require(${emissionType} == model.getEmissionType,
          s"initial model must be in the same type.")
        (model.initialProb, model.transitionProb)
      case None =>
        throw new UnknownError(s"Unsupervised training must specify an initial model")
    }

    ${emissionType} match {
      case "multinomial" =>
        val emissionProb = initialModel.get.asInstanceOf[MultinomialHMMModel].emissionProb
        var iteration = 0
        var isConverged = false
        var model = new MultinomialHMMModel(uid, initialProb, transitionProb, emissionProb)
          .setEmissionType("multinomial")
        while (iteration < getMaxIter && !isConverged) {
          val summarizer = {
            val seqOp = (c: BaumWelchSummarizer, instance: LabeledSequence) =>
              c.add(instance.features)
            val combOp = (c1: BaumWelchSummarizer, c2: BaumWelchSummarizer) =>
              c1.merge(c2)
            instances.treeAggregate(new BaumWelchSummarizer(model))(seqOp, combOp)
          }
          val newInitialProb = summarizer.initialProb
          val newTransitionProb = summarizer.transitionProb
          val newEmissionProb = summarizer.multinomialEmissionProb

          val diff = Vectors.sqdist(Vectors.dense(transitionProb.toArray),
            Vectors.dense(newTransitionProb.toArray))
          isConverged = diff < getTol
          model = new MultinomialHMMModel(uid, newInitialProb, newTransitionProb, newEmissionProb)
            .setEmissionType("multinomial")
          iteration += 1
        }
        model
      case _ =>
        throw new UnknownError(s"Unsupported emissionType: ${$(emissionType)}.")
    }
  }

  override def copy(extra: ParamMap): HiddenMarkovModel = defaultCopy(extra)
}

object HiddenMarkovModel extends DefaultParamsReadable[HiddenMarkovModel] {
  override  def load(path: String): HiddenMarkovModel = super.load(path)
}

/**
 * MultiClassSummarizer computes the number of distinct label transitions and corresponding counts,
 * and validates the data to see if the labels used for k state multi-state sequential tagging
 * are in the range of {0, 1, ..., k - 1} in an online fashion.
 *
 * @param smoothing
 */
private class MultiClassSummarizer(smoothing: Double) extends Serializable {
  private val distinctMap = new mutable.HashMap[Int, Long]
  private var totalInvalidCnt: Long = 0L
  private val firstLabelMap = new mutable.HashMap[Int, Long]
  private val transitionMap = new mutable.HashMap[(Int, Int), Long]

  def add(labels: Vector): this.type = {

    for (i <- 0 until labels.size) {
      val label = labels(i)
      if (label - label.toInt != 0.0 || label < 0) {
        totalInvalidCnt += 1
        this
      }
    }

    var srcLabel = 0.0
    for(i <- 0 until labels.size) {
      val label = labels(i)
      val counts: Long = distinctMap.getOrElse(label.toInt, 0L)
      distinctMap.put(label.toInt, counts + 1L)
      if (i == 0) {
        val firstCounts: Long = firstLabelMap.getOrElse(label.toInt, 0L)
        firstLabelMap.put(label.toInt, firstCounts + 1L)
      } else {
        val transCounts: Long = transitionMap.getOrElse((srcLabel.toInt, label.toInt), 0L)
        transitionMap.put((srcLabel.toInt, label.toInt), transCounts + 1L)
      }
      srcLabel = label
    }
    this
  }

  def merge(other: MultiClassSummarizer): MultiClassSummarizer = {
    val (largeMap, smallMap) = if (this.distinctMap.size > other.distinctMap.size) {
      (this, other)
    } else {
      (other, this)
    }
    smallMap.distinctMap.foreach {
      case (key, value) =>
        val counts: Long = largeMap.distinctMap.getOrElse(key, 0L)
        largeMap.distinctMap.put(key, counts + value)
    }
    smallMap.firstLabelMap.foreach {
      case (key, value) =>
        val counts: Long = largeMap.firstLabelMap.getOrElse(key, 0L)
        largeMap.firstLabelMap.put(key, counts + value)
    }
    smallMap.transitionMap.foreach {
      case (key, value) =>
        val counts: Long = largeMap.transitionMap.getOrElse(key, 0L)
        largeMap.transitionMap.put(key, counts + value)
    }
    largeMap.totalInvalidCnt += smallMap.totalInvalidCnt
    largeMap
  }

  /** @return The total invalid input counts. */
  def countInvalid: Long = totalInvalidCnt

  /** @return The number of distinct labels in the input dataset. */
  def numClasses: Int = if (distinctMap.isEmpty) 0 else distinctMap.keySet.max + 1

  def initialProb: Vector = {
    val result = Array.ofDim[Double](numClasses)
    val firstLabelSum = firstLabelMap.values.sum + numClasses * smoothing
    if (firstLabelSum > 0) {
      for (j <- 0 until numClasses) {
        result(j) = (firstLabelMap.getOrElse(j, 0L).toDouble + smoothing) / firstLabelSum.toDouble
      }
    }
    Vectors.dense(result)
  }

  def transitionProb: DenseMatrix = {
    val result = Array.ofDim[Double](numClasses * numClasses)
    for (i <- 0 until numClasses) {
      var sum = 0.0
      for (j <- 0 until numClasses) {
        sum += transitionMap.getOrElse((i, j), 0L).toDouble + smoothing
      }
      for (j <- 0 until numClasses) {
        result(i * numClasses + j) =
          (transitionMap.getOrElse((i, j), 0L).toDouble + smoothing) / sum
      }
    }
    new DenseMatrix(numClasses, numClasses, result)
  }
}

/**
 * MultinomialSummarizer computes the counts of distinct state-to-observation emissions
 * assuming that the state to observation emission probability is a multinomial distribution
 *
 * @param numClasses
 * @param smoothing
 */
private class MultinomialSummarizer(numClasses: Int, smoothing: Double) extends Serializable {
  private val distinctMap = new mutable.HashMap[Int, Long]
  private val emissionMap = new mutable.HashMap[(Int, Int), Long]
  private var totalInvalidCnt: Long = 0L

  def add(sequence: LabeledSequence): this.type = {
    val label = sequence.label
    val features = sequence.features
    require(label.size == features.numCols, s"Labels and features must have same number of frames.")
    var fi = 0
    features.colIter.foreach { fv =>
      val l = label(fi)
      if (l - l.toInt != 0.0 || l < 0 || l > numClasses - 1) {
        totalInvalidCnt += 1
      } else {
        val nonzero = fv.size match {
          case 1 => Option(fv(0))
          case _ => fv.toArray.filter(_ > 0.0).headOption
        }
        if (nonzero == None || nonzero.get.toInt - nonzero.get != 0.0 || nonzero.get < 0) {
          totalInvalidCnt += 1
        } else {
          val f = nonzero.get.toInt
          val counts: Long = distinctMap.getOrElse(f, 0L)
          distinctMap.put(f, counts + 1L)
          val emissionCnts: Long = emissionMap.getOrElse((l.toInt, f), 0L)
          emissionMap.put((l.toInt, f), emissionCnts + 1L)
        }
      }
      fi += 1
    }
    this
  }

  def merge(other: MultinomialSummarizer): MultinomialSummarizer = {
    val (largeMap, smallMap) = if (this.emissionMap.size > other.emissionMap.size) {
      (this, other)
    } else {
      (other, this)
    }
    smallMap.distinctMap.foreach {
      case (key, value) =>
        val counts: Long = largeMap.distinctMap.getOrElse(key, 0L)
        largeMap.distinctMap.put(key, counts + value)
    }
    smallMap.emissionMap.foreach {
      case (key, value) =>
        val counts: Long = largeMap.emissionMap.getOrElse(key, 0L)
        largeMap.emissionMap.put(key, counts + value)
    }
    largeMap
  }

  /** @return The total invalid input counts. */
  def countInvalid: Long = totalInvalidCnt

  /** @return The number of distinct labels in the input dataset. */
  def numFeatures: Int = if (distinctMap.isEmpty) 0 else distinctMap.keySet.max + 1

  def emissionProb: Matrix = {
    val result = Array.ofDim[Double](numFeatures * numClasses)
    for (i <- 0 until numClasses) {
      var sum = 0.0
      for (j <- 0 until numFeatures) {
        sum += emissionMap.getOrElse((i, j), 0L).toDouble + smoothing
      }
      for (j <- 0 until numFeatures) {
        result(i * numFeatures + j) = (emissionMap.getOrElse((i, j), 0L).toDouble + smoothing) / sum
      }
    }
    new DenseMatrix(numFeatures, numClasses, result)
  }
}

/**
 * EM training sufficient statistics summarizer using Baum-Welch algorithm
 *
 * @param model
 */
private class BaumWelchSummarizer(model: HMMModel) extends Serializable {

  private val numFeatures = model.numFeatures
  private val numClasses = model.initialProb.size
  private val xiSum = DenseMatrix.zeros(numClasses, numClasses)
  private val gammaSum = Vectors.zeros(numClasses)
  private val gammaFirstSum = Vectors.zeros(numClasses)
  private val activeGammaSum = DenseMatrix.zeros(numFeatures, numClasses)

  /**
   * E step for a single observation sequence
   *
   * @param features
   * @return
   */
  private[sequence] def add(features: Matrix): this.type = {
    val localXiSum = xiSum
    val localGammaSum = gammaSum
    val localGammaFirstSum = gammaFirstSum
    val localActiveGammaSum = activeGammaSum

    val emissions = model.precomputeEmissions(features)
    val alphas = model.forward(emissions)
    val betas = model.backward(emissions.tail)

    val margin = alphas.last.sum
    val firstSum = (alphas.head, betas.head).zipped.map((a, b) => a * b)
    BLAS.axpy(1.0 / margin, Vectors.dense(firstSum), localGammaFirstSum)

    var prevAlpha = Array.fill[Double](numClasses)(0.0)
    val gammas = (alphas, betas, emissions).zipped.foldLeft(List[Array[Double]]()) { (r, c) =>
      val (alpha, beta, emission) = c
      val alphaBetaProd = (alpha, beta).zipped.map((a, b) => a * b)
      val gamma = alphaBetaProd.map(_ / margin)
      BLAS.axpy(1.0, Vectors.dense(gamma), localGammaSum)
      val xi = DenseMatrix.diag(Vectors.dense((beta, emission).zipped.map((b, e) => b * e)))
        .multiply(model.transitionProb)
        .multiply(DenseMatrix.diag(Vectors.dense(prevAlpha)))
      BLAS.axpy(1.0 / margin, xi, localXiSum)
      prevAlpha = alpha
      gamma :: r
    }.reverse

    BLAS.axpy(1.0, model.getEmissionStats(features, gammas), localActiveGammaSum)

    this
  }

  def merge(other: BaumWelchSummarizer): this.type = {
    BLAS.axpy(1.0, other.xiSum, xiSum)
    BLAS.axpy(1.0, other.gammaSum, gammaSum)
    BLAS.axpy(1.0, other.gammaFirstSum, gammaFirstSum)
    BLAS.axpy(1.0, other.activeGammaSum, activeGammaSum)
    this
  }

  def initialProb: Vector = {
    val initialSum = gammaFirstSum.toArray.sum
    require(initialSum > 0, s"Sum of initial occurrences must be positive.")
    Vectors.dense(gammaFirstSum.toArray.map( _ / initialSum))
  }

  def transitionProb: DenseMatrix = {
    val result = xiSum.colIter.flatMap { v => v.toArray.map(_ / v.toArray.sum)}.toArray
    new DenseMatrix(numClasses, numClasses, result)
  }

  def multinomialEmissionProb: Matrix = {
    val result = (activeGammaSum.colIter zip gammaSum.toArray.iterator).toArray.flatMap {
      case (x, y) => x.toArray.map( _ / y)
    }
    new DenseMatrix(numFeatures, numClasses, result)
  }
}

/**
 * Define common algorithms including forward, backward and viterbi for HMM models
 *
 * @param uid
 * @param initialProb initial probability vector
 * @param transitionProb state transition matrix
 */
abstract class HMMModel private[ml] (
    val uid: String,
    val initialProb: Vector,
    val transitionProb: DenseMatrix) extends MarginalTaggingModel[Matrix, HMMModel]
  with HiddenMarkovModelParams with MLWritable{

  override val numClasses: Int = transitionProb.numRows

  def setEmissionType(value: String): this.type = set(emissionType, value)
  setDefault(emissionType -> "multinomial")

  val transitionProbTransposed: DenseMatrix = transitionProb.transpose

  /**
   * Compute feature scores for all classes in all frames
   *
   * @param features
   * @return
   */
  def precomputeEmissions(features: Matrix): List[Array[Double]]

  /**
   * Accumulate sufficient statistics for emission model depending on the emission type
   *
   * @param features
   * @param gammas
   * @return
   */
  def getEmissionStats(features: Matrix, gammas: List[Array[Double]]): DenseMatrix

  /**
   * Iterative forward algorithm
   *
   * @param emissions
   * @return
   */
  def forward(emissions: Traversable[Array[Double]]): List[Array[Double]] = {
    emissions.foldLeft(List[Array[Double]]()) { (r, c) =>
      r match {
        case head :: tail =>
          val transitionSum = transitionProb.multiply(Vectors.dense(head))
          (c, transitionSum.toArray).zipped.map((e, t) => e * t) :: r
        case Nil =>
          val alphaZero = (c, initialProb.toArray).zipped.map((e, t) => e * t)
          alphaZero :: r
      }
    }.reverse
  }

  /**
   * Iterative backward algorithm
   *
   * @param emissions
   * @return
   */
  def backward(emissions: Traversable[Array[Double]]): List[Array[Double]] = {
    emissions.foldRight(List[Array[Double]](Array.fill(numClasses)(1.0))) { (c, r) =>
      val emissionBetaProd = (c, r.head).zipped.map((e, b) => e * b)
      val transitionSum = transitionProbTransposed.multiply(Vectors.dense(emissionBetaProd))
      transitionSum.toArray :: r
    }
  }

  /**
   * Get the marginal/posterior probability of the observation sequence
   *
   * @param features
   * @return margin score
   */
  override def getMargin(features: Matrix): Double = {
    val emissions = precomputeEmissions(features)
    val forwards = forward(emissions)
    forwards.last.sum
  }

  /**
   * Viterbi decoding
   *
   * @param features observation sequence in the form of a matrix(F, T)s
   * @return list of [[Tuple2]]s of Double and vector where
   *         k-th Double is the confidence score for the k-th vector
   *         which is the k-th highest confident label sequence.
   */
  override protected def decodeRaw(features: Matrix): Array[(Double, Vector)] = {
    val emissions = precomputeEmissions(features)
    var numFrames = 0

    // run Viterbi and score back pointers for observations 1 ... T-1
    var prevAlpha = initialProb.toArray
    val viterbi = emissions.foldLeft(List[Array[Int]]()) { (r, c) =>
      numFrames += 1
      r match {
        case head :: tail =>
          val productProb = Matrices.diag(Vectors.dense(prevAlpha))
            .multiply(transitionProb)
            .multiply(DenseMatrix.diag(Vectors.dense(c)))
          val (currAlpha, backs) = productProb.toArray.grouped(productProb.numRows).map {
            _.zipWithIndex.sortWith((x, y) => x._1 > y._1).head
          }.toArray.unzip
          prevAlpha = currAlpha
          backs :: r
        case Nil =>
          // initialize alpha as the initial probabilities for each class, i.e. pi(k)
          prevAlpha = (c, initialProb.toArray).zipped.map((e, t) => e * t)
          Array.fill[Int](numClasses)(-1) :: r
      }

    }.dropRight(1)

    // back tracking to generate optimal label sequences that end in each class
    val label = viterbi.foldLeft(List[Array[Int]]((0 until numClasses).toArray)) { (r, c) =>
      r.head.map(c(_)) :: r
    }

    // generate results
    val raw = label.flatMap(_.map(_.toDouble)).toArray
    val rawMatrix = Matrices.dense(numClasses, numFrames, raw)

    val result = new Array[(Double, Vector)](numClasses)
    rawMatrix.rowIter.zipWithIndex.foreach { case (r, i) => result(i) = (prevAlpha(i), r)}
    result.sortWith((x, y) => x._1 > y._1)
  }

  override protected def raw2prediction(raw: Array[(Double, Vector)]): Vector = {
    raw(Vectors.dense(raw.map(_._1)).argmax)._2
  }

}

/**
 * HMM model with Multinomial emission, i.e.
 * observation at each frame is a random feature index generated from a Multinomial distribution
 *
 * @param uid
 * @param initialProb initial probability vector
 * @param transitionProb state transition matrix
 * @param emissionProb state emission matrix
 */
class MultinomialHMMModel private[ml](
    uid: String,
    initialProb: Vector,
    transitionProb: DenseMatrix,
    val emissionProb: Matrix) extends HMMModel(uid, initialProb, transitionProb)
  with HiddenMarkovModelParams with MLWritable {

  override val numFeatures: Int = emissionProb.numRows

  protected def getFrameEmission(fv: Double): Array[Double] = {
    require(fv.toInt.toDouble == fv && fv.toInt < numFeatures,
      s"Features must be integers representing in the range of (0, ${numFeatures})")
    val i = fv.toInt
    val result = Array.ofDim[Double](numClasses)
    for (j <- 0 until numClasses) {
      result(j) = emissionProb(i, j)
    }
    result
  }

  override def precomputeEmissions(features: Matrix): List[Array[Double]] = {
    val featVector = checkInputFeature(features)
    require(featVector.size > 0, s"Sequence should be longer than zero.")
    featVector.toArray.map(fv => getFrameEmission(fv)).toList
  }

  override def getEmissionStats(features: Matrix, gammas: List[Array[Double]]): DenseMatrix = {
    val featVector = checkInputFeature(features)
    require(featVector.size == gammas.size, s"Incompatible emissions.")
    val result = Array.fill[Double](numFeatures * numClasses)(0.0)
    var fi = 0
    gammas.foreach { g =>
      val i = featVector(fi).toInt
      for (j <- 0 until numClasses) {
        result(j * numFeatures + i) += g(j)
      }
      fi += 1
    }
    new DenseMatrix(numFeatures, numClasses, result)
  }

  override def write: MLWriter = new MultinomialHMMModel.HMMModelWriter(this)

  override def copy(extra: ParamMap): MultinomialHMMModel = {
    copyValues(new MultinomialHMMModel(uid, initialProb, transitionProb, emissionProb), extra)
  }

  /**
   * Check the raw feature input and convert it to a vector, i.e. 1 * T matrix
   *
   * @param features
   * @return feature vector
   */
  protected def checkInputFeature(features: Matrix): Vector = {
    val values = features.toArray
    if (!values.forall(f => f.toInt.toDouble == f && f.toInt < numFeatures)) {
      throw new SparkException(
        s"Features must be integers representing in the range of (0, ${numFeatures})")
    }
    if (features.numRows == 1 || features.numCols == 1) {
      Vectors.dense(values)
    } else if (features.numRows != numFeatures) {
      throw new SparkException(
        s"Matrix features must have row number equal to ${numFeatures})")
    } else {
      if (!values.grouped(numFeatures).forall(v => v.count(_ != 0) == 1)) {
        throw new SparkException(s"Each frame must contain one and only one nonzero element.")
      } else {
        val results = values.grouped(numFeatures).flatMap(_.filter(v => v != 0)).toArray
        Vectors.dense(results)
      }
    }
  }
}

object MultinomialHMMModel extends MLReadable[MultinomialHMMModel] {

  override def read: MLReader[MultinomialHMMModel] = new HMMModelReader

  override def load(path: String): MultinomialHMMModel = super.load(path)

  private[MultinomialHMMModel] class HMMModelWriter(instance: MultinomialHMMModel)
    extends MLWriter with Logging {

    private case class Data(initialProb: Vector, transitionProb: DenseMatrix, emissionProb: Matrix)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: initialProb, transitionProb, emissionProb
      val data = Data(instance.initialProb, instance.transitionProb, instance.emissionProb)
      val dataPath = new Path(path, "data").toString
      sqlContext.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class HMMModelReader extends MLReader[MultinomialHMMModel] {

    private val className = classOf[MultinomialHMMModel].getName

    override def load(path: String): MultinomialHMMModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sqlContext.read.format("parquet").load(dataPath)
        .select("initialProb", "transitionProb", "emissionProb").head()
      val initialProb = data.getAs[Vector](0)
      val transitionProb = data.getAs[DenseMatrix](1)
      val emissionProb = data.getAs[Matrix](2)
      val model = new MultinomialHMMModel(metadata.uid, initialProb, transitionProb, emissionProb)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
