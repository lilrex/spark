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

package org.apache.spark.ml

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.linalg.{Matrix, MatrixUDT, Vector, VectorUDT}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.sequence.LabeledSequence
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}

/**
 * (private[ml]) Trait for parameters for decoding (sequence).
 */
private[ml] trait DecoderParams extends Params
  with HasLabelCol with HasFeaturesCol with HasPredictionCol {

  /**
   * Validates and transforms the input schema with the provided param map.
   *
   * @param schema           input schema
   * @param fitting          whether this is in fitting
   * @param featuresDataType SQL DataType for FeaturesType.
   *                         E.g., [[org.apache.spark.mllib.linalg.VectorUDT]]
   *                         for vector features.
   * @return output schema
   */
  protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType,
      labelDataType: DataType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), featuresDataType)
    if (fitting) {
      SchemaUtils.checkColumnType(schema, $(labelCol), labelDataType)
    }
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}

/**
 * :: DeveloperApi ::
 * Abstraction for decoding problems (tagging and segmentation).
 * @tparam FeaturesType  Type of features.
 *                       E.g., [[org.apache.spark.mllib.linalg.MatrixUDT]] for matrix features.
 * @tparam Learner  Specialization of this class.  If you subclass this type, use this type
 *                  parameter to specify the concrete type.
 * @tparam M  Specialization of [[DecodingModel]].  If you subclass this type, use this type
 *            parameter to specify the concrete type for the corresponding model.
 *
 */
@DeveloperApi
abstract class Decoder[
    FeaturesType,
    Learner <: Decoder[FeaturesType, Learner, M],
    M <: DecodingModel[FeaturesType, M]]
  extends Estimator[M] with DecoderParams {

  /** @group setParam */
  def setLabelCol(value: String): Learner = set(labelCol, value).asInstanceOf[Learner]

  /** @group setParam */
  def setFeatureCol(value: String): Learner = set(labelCol, value).asInstanceOf[Learner]

  /** @group setParam */
  def setPredictionCol(value: String): Learner = set(predictionCol, value).asInstanceOf[Learner]

  override def fit(dataset: Dataset[_]): M = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)
    copyValues(train(dataset).setParent(this))
  }

  override def copy(extra: ParamMap): Learner

  /**
   * Train a model using the given dataset and parameters.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset Training dataset
   * @return Fitted model
   */
  protected def train(dataset: Dataset[_]): M

  /**
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is MatrixUDT, but it may be overridden if FeaturesType is not Matrix.
   */
  private[ml] def featuresDataType: DataType = new MatrixUDT

  private[ml] def labelDataType: DataType = new VectorUDT

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = true, featuresDataType, labelDataType)
  }

  /**
   * Extract [[labelCol]] and [[featuresCol]] from the given dataset,
   * and put it in an RDD with strong types.
   */
  protected def extractLabeledSequences(dataset: Dataset[_]): RDD[LabeledSequence] = {
    val l = if (!isDefined(labelCol) || $(labelCol).isEmpty) lit(null) else col($(labelCol))
    dataset.select(l, col($(featuresCol))).rdd.map {
      case Row(label: Vector, features: Matrix) => LabeledSequence(label, features)
      case Row(null, features: Matrix) => LabeledSequence(null, features)
    }
  }
}

/**
 * :: DeveloperApi ::
 * Abstraction for a model for decoding tasks (tagging and segmentation).
 *
 * @tparam FeaturesType Type of features.
 *                     E.g., [[org.apache.spark.mllib.linalg.MatrixUDT]] for matrix features.
 * @tparam M specialization of [[DecodingModel]]. If you subclass this type, use this type
 *           parameter to specify the concrete type of the corresponding model.
 */
@DeveloperApi
abstract class DecodingModel[FeaturesType, M <: DecodingModel[FeaturesType, M]]
  extends Model[M] with DecoderParams {

  /** @group setParam */
  def setFeaturesCol(value: String): M = set(featuresCol, value).asInstanceOf[M]

  /** @group setParam */
  def setPredictionCol(value: String): M = set(predictionCol, value).asInstanceOf[M]

  /** Returns the number of features the model was trained on. If unknown, returns -1 */
  def numFeatures: Int = -1

  /**
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is MatrixUDT, but it may be overridden if FeaturesType is not Matrix.
   */
  protected def featuresDataType: DataType = new MatrixUDT

  protected def labelDataType: DataType = new VectorUDT

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = false, featuresDataType, labelDataType)
  }

  /**
   * Transforms dataset by reading from [[featuresCol]], calling [[decode()]], and storing
   * the predictions as a new column [[predictionCol]].
   *
   * @param dataset input dataset
   * @return transformed dataset with [[predictionCol]] of type [[Vector]]
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    if ($(predictionCol).nonEmpty) {
      transformImpl(dataset)
    } else {
      this.logWarning(s"$uid: Decoder.transform() was called as NOOP" +
        " since no output columns were set.")
      dataset.toDF()
    }
  }

  protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf { (features: Any) =>
      decode(features.asInstanceOf[FeaturesType])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  /**
   * Decode sequence of labels for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  protected def decode(features: FeaturesType): Vector
}
