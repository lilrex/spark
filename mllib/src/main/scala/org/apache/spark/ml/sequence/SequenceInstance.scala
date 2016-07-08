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

import org.apache.spark.ml.linalg.{Matrix, Vector}

/**
 * Class that represent an instance of weighted data point with sequential labels and features.
 *
 * @param labels Labels for this sequence
 * @param weight The weight of this instance
 * @param features The matrix of features for this sequence, i.e. a feature vector for each label
 */
private[ml] case class SequenceInstance(labels: Vector, weight: Double, features: Matrix)
