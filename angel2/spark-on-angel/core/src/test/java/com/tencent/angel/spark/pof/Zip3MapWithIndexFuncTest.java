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

package com.tencent.angel.spark.pof;

import com.tencent.angel.ml.matrix.psf.update.enhance.zip3.Zip3MapWithIndexFunc;
import io.netty.buffer.ByteBuf;


public class Zip3MapWithIndexFuncTest implements Zip3MapWithIndexFunc {

  private double multiplier;

  public Zip3MapWithIndexFuncTest(double multiplier) {
    this.multiplier = multiplier;
  }

  public Zip3MapWithIndexFuncTest() {
  }

  @Override
  public double call(int index, double value1, double value2, double value3) {
    if (index == 0) {
      return value1 * (1.0 - value3) + value2 * value2;
    } else {
      return multiplier * value1 * (1.0 - value3) + value2 * value2;
    }
  }

  @Override
  public void serialize(ByteBuf buf) {
    buf.writeDouble(multiplier);
  }

  @Override
  public void deserialize(ByteBuf buf) {
    multiplier = buf.readDouble();
  }

  @Override
  public int bufferLen() {
    return 8;
  }

}
