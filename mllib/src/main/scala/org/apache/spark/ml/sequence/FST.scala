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

trait FST[Q, Sigma, Gamma] {

  def transitionsFrom(q: Q): Seq[Transition[Q, Sigma, Gamma]]

  def transitionsTo(r: Q): Seq[Transition[Q, Sigma, Gamma]]
}

object FST {

  def apply[Q, Sigma, Gamma](initialStates: Set[Q], finalStates: Set[Q])
                            (transitions: (Q, (Q, Sigma, Gamma))*): FST[Q, Sigma, Gamma] = {
    val deltas = for ((from, (to, in, out)) <- transitions) yield Transition(from, to, in, out);
    val allTransitionsByOrigin = deltas.groupBy(_.q);

    val allTransitionsByTarget = deltas.groupBy(_.r);

    new FST[Q, Sigma, Gamma] {
      def transitionsFrom(q: Q) = allTransitionsByOrigin.getOrElse(q, Seq.empty)

      def transitionsTo(r: Q) = allTransitionsByTarget.getOrElse(r, Seq.empty)
    }
  }
}

case class Transition[Q, Sigma, Gamma](q: Q, r: Q, a: Sigma, b: Gamma);
