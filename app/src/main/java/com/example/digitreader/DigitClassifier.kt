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
        interpreter.allocateTensors()
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

    fun predict(input: FloatArray): FloatArray {
        val x = toBatch28x28(input)
        val out = arrayOf(FloatArray(10))
        val inKey = interpreter.getSignatureInputs("infer").firstOrNull() ?: "x"
        val outKey = interpreter.getSignatureOutputs("infer").firstOrNull() ?: "output"
        interpreter.runSignature(
            mapOf(inKey to x),
            mapOf(outKey to out),
            "infer"
        )
        return out[0]
    }

    fun train(input: FloatArray, correctDigit: Int): Float {
        val x = toBatch28x28(input)
        val y = arrayOf(FloatArray(10).also { it[correctDigit] = 1f })
        val lossOut = FloatArray(1)
        val outKey = interpreter.getSignatureOutputs("train").firstOrNull() ?: "loss"

        interpreter.runSignature(
            mapOf("x" to x, "y" to y),
            mapOf(outKey to lossOut),
            "train"
        )
        return lossOut[0]
    }

    fun saveWeights(path: String) {
        val inKey = interpreter.getSignatureInputs("save").firstOrNull() ?: "checkpoint_path"
        val outKey = interpreter.getSignatureOutputs("save").firstOrNull() ?: "checkpoint_path"
        val out = arrayOf("")
        interpreter.runSignature(
            mapOf(inKey to arrayOf(path)),
            mapOf(outKey to out),
            "save"
        )
    }

    fun restoreWeightsIfExists(path: String) {
        if (!File(path).exists()) return
        val inKey = interpreter.getSignatureInputs("restore").firstOrNull() ?: "checkpoint_path"
        val outputs = mutableMapOf<String, Any>()
        for (outName in interpreter.getSignatureOutputs("restore")) {
            val tensor = interpreter.getOutputTensorFromSignature(outName, "restore")
            val buf = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
            outputs[outName] = buf
        }
        interpreter.runSignature(mapOf(inKey to arrayOf(path)), outputs, "restore")
    }

    fun close() = interpreter.close()
}
