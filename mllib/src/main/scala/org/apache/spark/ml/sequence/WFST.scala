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

/**
 * Weighted Finite State Transducer
 *
 * @tparam Q      State type
 * @tparam Sigma  Input type
 * @tparam Gamma  Output type
 * @tparam W      Weight type
 */
private[spark] trait WFST[Q, Sigma, Gamma, W] {

  def transitionsFrom(q: Q): Seq[WeightedTransition[Q, Sigma, Gamma, W]]

  def transitionsTo(r: Q): Seq[WeightedTransition[Q, Sigma, Gamma, W]]

  def initialStateWeights: Map[Q, W]

  def finalStateWeights: Map[Q, W]
}

object WFST {

  def apply[Q, Sigma, Gamma, W](
     initialStates: Set[Q], finalStates: Set[Q])
     (initialWeights: Map[Q, W], finalWeights: Map[Q, W])
     (transitions: (Q, (Q, Sigma, Gamma, W))*): WFST[Q, Sigma, Gamma, W] = {

    val deltas = for ((from, (to, in, out, w)) <- transitions)
      yield WeightedTransition(from, to, in, out, w)

    val allTransitionsByOrigin = deltas.groupBy(_.q)

    val allTransitionsByTarget = deltas.groupBy(_.r)

    new WFST[Q, Sigma, Gamma, W] {
      def transitionsFrom(q: Q) = allTransitionsByOrigin.getOrElse(q, Seq.empty)

      def transitionsTo(r: Q) = allTransitionsByTarget.getOrElse(r, Seq.empty)

      val initialStateWeights = initialWeights

      val finalStateWeights = finalWeights
    }
  }
}

case class WeightedTransition[Q, Sigma, Gamma, W](q: Q, r: Q, a: Sigma, b: Gamma, w: W)
