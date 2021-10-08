/*
 * https://github.com/android/camera-samples/tree/main/CameraXTfLite
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.asltflite

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import kotlin.collections.*

/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class ObjectDetectionHelper(private val tflite: Interpreter, private val labels: List<String>) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    //data class ObjectPrediction(val location: RectF, val label: String, val score: Float)
    data class ObjectPrediction2(val label: String, val score: Float)

    //private val locations = arrayOf(Array(OBJECT_COUNT) { FloatArray(4) })
    private val labelIndices = arrayOf(FloatArray(OBJECT_COUNT))
    private val scores = arrayOf(FloatArray(OBJECT_COUNT))


    // The original example included view box location data and an output buffer that was not
    // easily compatible with our .tflite model output
    /**
    private val outputBuffer = mapOf(
        0 to locations,
        1 to labelIndices,
        2 to scores,
        3 to FloatArray(1)
    ) */

    private val outputBuffer2 = mapOf(
        0 to scores
    )


    //TODO fix the mapping of labels to scores
    private val predictions2 get() = (0 until OBJECT_COUNT).map {
            ObjectPrediction2(
                label = labels[labelIndices[0][it].toInt()],
                score = scores[0][it]
            )

    }

    fun predict2(image: TensorImage): List<ObjectPrediction2> {
        tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer2)
        return predictions2
    }
    companion object {
        const val OBJECT_COUNT = 28
    }

}





