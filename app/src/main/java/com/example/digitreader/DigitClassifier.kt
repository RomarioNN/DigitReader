package com.example.digitreader

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DigitClassifier(context: Context) {

    private val interpreter: Interpreter

    init {
        val buffer = loadModel(context)
        interpreter = Interpreter(buffer)
    }

    private fun loadModel(context: Context): ByteBuffer {
        val bytes = context.assets.open("mnist_trainable.tflite").readBytes()
        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .also { it.put(bytes); it.rewind() }
    }

    private fun toBatch28x28(flat: FloatArray): Array<Array<FloatArray>> {
        return Array(1) {
            Array(28) { r ->
                FloatArray(28) { c -> flat[r * 28 + c] }
            }
        }
    }

    /** input: FloatArray(784), нормализованный в [0,1] */
    fun predict(input: FloatArray): FloatArray {
        val x = toBatch28x28(input)                 // [1, 28, 28]
        val out = arrayOf(FloatArray(10))           // [1, 10]
        interpreter.runSignature(
            mapOf("x" to x),
            mapOf("output" to out),
            "infer"
        )
        return out[0]
    }

    /** Дообучение на одном примере. correctDigit — правильный класс 0..9 */
    fun train(input: FloatArray, correctDigit: Int): Float {
        val x = toBatch28x28(input)                 // [1, 28, 28]
        val y = arrayOf(FloatArray(10).also { it[correctDigit] = 1f })
        val lossOut = arrayOf(FloatArray(1))

        interpreter.runSignature(
            mapOf("x" to x, "y" to y),
            mapOf("loss" to lossOut),
            "train"
        )
        return lossOut[0][0]
    }

    fun saveWeights(path: String) {
        val inp = mapOf("checkpoint_path" to arrayOf(path))
        interpreter.runSignature(inp, emptyMap(), "save")
    }

    fun restoreWeightsIfExists(path: String) {
        if (!File(path).exists()) return
        val inp = mapOf("checkpoint_path" to arrayOf(path))
        interpreter.runSignature(inp, emptyMap(), "restore")
    }

    fun close() = interpreter.close()
}
