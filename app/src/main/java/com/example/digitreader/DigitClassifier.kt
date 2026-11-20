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
        initWeights()
    }

    private fun loadModel(context: Context): ByteBuffer {
        val bytes = context.assets.open("mnist_trainable.tflite").readBytes()
        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .also { it.put(bytes); it.rewind() }
    }

    fun initWeights() {
        val sigs = interpreter.signatureKeys
        if (sigs.contains("init_weights")) {
            val inKey = interpreter.getSignatureInputs("init_weights").firstOrNull() ?: "trigger"
            val outKey = interpreter.getSignatureOutputs("init_weights").firstOrNull() ?: "ok"
            val out = FloatArray(1)
            interpreter.runSignature(
                mapOf(inKey to floatArrayOf(1f)),
                mapOf(outKey to out),
                "init_weights"
            )
        }
    }

    private fun toBatch28x28(inputs: List<FloatArray>): Array<Array<FloatArray>> {
        return Array(inputs.size) { b ->
            val flat = inputs[b]
            Array(28) { r ->
                FloatArray(28) { c -> flat[r * 28 + c] }
            }
        }
    }

    fun predict(input: FloatArray): FloatArray {
        val x = toBatch28x28(listOf(input))
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

    /** Батчевое дообучение (Experience Replay): обучает одновременно на новом и прошлых примерах */
    fun trainBatch(inputs: List<FloatArray>, labels: List<Int>, epochs: Int = 5): Float {
        if (inputs.isEmpty()) return 0f
        val x = toBatch28x28(inputs)
        val y = Array(labels.size) { idx ->
            FloatArray(10).also { it[labels[idx].coerceIn(0, 9)] = 1f }
        }
        val lossOut = FloatArray(1)
        val outKey = interpreter.getSignatureOutputs("train").firstOrNull() ?: "loss"

        var lastLoss = 0f
        repeat(epochs.coerceAtLeast(1)) {
            interpreter.runSignature(
                mapOf("x" to x, "y" to y),
                mapOf(outKey to lossOut),
                "train"
            )
            lastLoss = lossOut[0]
        }
        return lastLoss
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
