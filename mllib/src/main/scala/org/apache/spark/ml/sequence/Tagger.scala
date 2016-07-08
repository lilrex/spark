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

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.{Decoder, DecoderParams, DecodingModel}
import org.apache.spark.ml.linalg.{Matrix, MatrixUDT, Vector}
import org.apache.spark.ml.param.shared.HasRawPredictionCol
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, StructType}

/**
 * [private[spark]] Params for tagging.
 */
private[spark] trait TaggerParams
  extends DecoderParams with HasRawPredictionCol {

  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType,
      labelDataType: DataType): StructType = {
    val parentSchema = super.
      validateAndTransformSchema(schema, fitting, featuresDataType, labelDataType)
    SchemaUtils.appendColumn(parentSchema, $(rawPredictionCol), new MatrixUDT)
  }
}


/**
 * :: DeveloperApi ::
 *
 * Sequential tagger.
 * Class Labels for each element are indexed {0, 1, ..., numClasses - 1}.
 *
 * @tparam FeaturesType Type of input features. E.g., [[Matrix]]
 * @tparam E            Concrete Estimator type
 * @tparam M            Concrete Model type
 */
@DeveloperApi
abstract class Tagger [
    FeaturesType,
    E <: Tagger[FeaturesType, E, M],
    M <: TaggingModel[FeaturesType, M]]
  extends Decoder[FeaturesType, E, M] with TaggerParams {

  /** @group setParam */
  def setRawPredictionCol(value: String): E = set(rawPredictionCol, value).asInstanceOf[E]

}

/**
 * :: DeveloperApi ::
 *
 * Model produced by a [[Tagger]].
 * Class Labels for each element are indexed {0, 1, ..., numClasses - 1}.
 *
 * @tparam FeaturesType Type of input features. E.g., [[Matrix]]
 * @tparam M            Concrete Model type
 */
@DeveloperApi
abstract class TaggingModel[FeaturesType, M <: DecodingModel[FeaturesType, M]]
  extends DecodingModel[FeaturesType, M] with TaggerParams {

  /** @group setParam */
  def setRawPredictionCol(value: String): M = set(rawPredictionCol, value).asInstanceOf[M]

  /** Number of classes (values which the label can take). */
  def numClasses: Int

  /**
   * Transforms dataset by reading from [[featuresCol]], and appending new columns as specified by
   * parameters:
   * - predicted labels as [[predictionCol]] of type [[Vector]]
   * - raw predictions (confidences) as [[rawPredictionCol]] of type [[Matrix]].
   *
   * @param dataset input dataset
   * @return transformed dataset
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = dataset
    var numColsOutput = 0
    if (getRawPredictionCol != "") {
      val predictRawUDF = udf { (features: Any) =>
        decodeRaw(features.asInstanceOf[FeaturesType])
      }
      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
      numColsOutput += 1
    }
    if (getPredictionCol != "") {
      val predUDF = if (getRawPredictionCol != "") {
        udf(raw2prediction _).apply(col(getRawPredictionCol))
      } else {
        val predictUDF = udf { (features: Any) =>
          decode(features.asInstanceOf[FeaturesType])
        }
        predictUDF(col(getFeaturesCol))
      }
      outputData = outputData.withColumn(getPredictionCol, predUDF)
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      logWarning(s"$uid: ClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF()
  }

  /**
   * Decode 1-best label sequence for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   *
   * This default implementation for taggers decode
   */
  override protected def decode(features: FeaturesType): Vector = {
    raw2prediction(decodeRaw(features))
  }

  /**
   * Raw decoding for top n label sequence
   * The meaning of a "raw" prediction may vary between algorithms, but it intuitively gives
   * a measure of confidence in each possible label sequence (where larger = more confident).
   * This internal method is used to implement [[transform()]] and output [[rawPredictionCol]].
   *
   * @return list of [[Tuple2]]s of Double and vector where
   *         k-th Double is the confidence score for the k-th vector
   *         which is the k-th highest confident label sequence.
   */
  protected def decodeRaw(features: FeaturesType): Array[(Double, Vector)]

  /**
   * Given an array of raw predictions, select the predicted label sequence.
   * This may be overridden to support threshold pruning.
   * @return predicted label sequence
   */
  protected def raw2prediction(rawPrediction: Array[(Double, Vector)]): Vector =
    rawPrediction.head._2

}
