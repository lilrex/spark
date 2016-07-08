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
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.param.shared.{HasProbabilityCol, HasThreshold}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}

/**
 * (private[sequence]) Params for probabilistic sequence-tagging.
 */
private[sequence] trait MarginalTaggerParams
  extends TaggerParams with HasProbabilityCol with HasThreshold {
  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType,
      labelDataType: DataType): StructType = {
    val parentSchema = super.
      validateAndTransformSchema(schema, fitting, featuresDataType, labelDataType)
    SchemaUtils.appendColumn(parentSchema, $(probabilityCol), DoubleType)
  }
}

/**
 * :: DeveloperApi ::
 *
 * Sequential tagger which can output posterior conditional probabilities for
 * label-sequence candidates
 *
 * @tparam FeaturesType Type of input features. E.g., [[Matrix]]
 * @tparam E            Concrete Estimator type
 * @tparam M            Concrete Model type
 */
@DeveloperApi
abstract class MarginalTagger [
    FeaturesType,
    E <: MarginalTagger[FeaturesType, E, M],
    M <: MarginalTaggingModel[FeaturesType, M]]
  extends Tagger[FeaturesType, E, M] with MarginalTaggerParams {

  /** @group setParam */
  def setProbabilityCol(value: String): E = set(probabilityCol, value).asInstanceOf[E]

  /** @group setParam */
  def setThreshold(value: Double): E = set(threshold, value).asInstanceOf[E]
}


/**
 * :: DeveloperApi ::
 *
 * Model produced by a [[MarginalTagger]].
 * Class Labels for each element are indexed {0, 1, ..., numClasses - 1}.
 *
 * @tparam FeaturesType Type of input features. E.g., [[Matrix]]
 * @tparam M            Concrete Model type
 */
@DeveloperApi
abstract class MarginalTaggingModel[
    FeaturesType,
    M <: MarginalTaggingModel[FeaturesType, M]]
 extends TaggingModel[FeaturesType, M] with MarginalTaggerParams {

  /** @group setParam */
  def setProbabilityCol(value: String): M = set(probabilityCol, value).asInstanceOf[M]

  /** @group setParam */
  def setThresholds(value: Double): M = set(threshold, value).asInstanceOf[M]

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = dataset
    var numColsOutput = 0
    if ($(rawPredictionCol).nonEmpty) {
      val predictRawUDF = udf { (features: Any) =>
        decode(features.asInstanceOf[FeaturesType])
      }
      outputData = outputData.withColumn($(rawPredictionCol), predictRawUDF(col($(featuresCol))))
      numColsOutput += 1
    }
    if ($(probabilityCol).nonEmpty) {
      val probUDF = udf { (features: Any) =>
          getMargin(features.asInstanceOf[FeaturesType])
      }
      outputData = outputData.withColumn($(probabilityCol), probUDF(col($(featuresCol))))
      numColsOutput += 1
    }
    if ($(predictionCol).nonEmpty) {
      val predUDF = udf { (features: Any) =>
        decode(features.asInstanceOf[FeaturesType])
      }
      outputData = outputData.withColumn($(predictionCol), predUDF(col($(featuresCol))))
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: ProbabilisticTaggingModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF()
  }

  /**
   * Calculate the probability of each label sequence given the features.
   * These probabilities are also called posterior probabilities.
   *
   * This internal method is used to implement [[transform()]] and output [[probabilityCol]].
   *
   * @return Estimated probabilities
   */
  protected def getMargin(features: FeaturesType): Double

}
