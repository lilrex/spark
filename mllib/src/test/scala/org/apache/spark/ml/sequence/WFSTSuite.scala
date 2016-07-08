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

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext

class WFSTSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("WFST Initialization") {
    val wfst = WFST[Int, String, String, Double](
      Set(0), Set(3))(
      Map(0 -> 1.0), Map(3 -> 0.6))(
      0 -> (1, "a", "b", 0.1),
      0 -> (2, "b", "a", 0.2),
      1 -> (1, "c", "a", 0.3),
      1 -> (3, "a", "a", 0.4),
      2 -> (3, "b", "b", 0.5)
    )

    val isw = wfst.initialStateWeights
    assert(isw.size == 1 && isw.head._1 == 0 && isw.head._2 == 1.0)

    val fsw = wfst.finalStateWeights
    assert(fsw.size == 1 && fsw.head._1 == 3 && fsw.head._2 == 0.6)

    val fromInitial = wfst.transitionsFrom(0)
    assert(fromInitial.size == 2 && fromInitial.head.r == 1 && fromInitial(1).r == 2)

    val toFinal = wfst.transitionsTo(3)
    assert(toFinal.size == 2 && toFinal.head.q == 1 && toFinal(1).q == 2)
  }
}
