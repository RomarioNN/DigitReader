package com.example.digitreader

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var capturedImage: ImageView
    private lateinit var resultDigit: TextView
    private lateinit var confidenceText: TextView
    private lateinit var captureBtn: Button
    private lateinit var retryBtn: Button
    private lateinit var wrongBtn: Button

    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    private var classifier: DigitClassifier? = null
    private var modelInitError: String? = null
    private var currentInput: FloatArray? = null
    private var currentDigit: Int = -1

    private val checkpointFile: File by lazy {
        File(filesDir, "digit_weights.ckpt")
    }

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Нужна камера", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView     = findViewById(R.id.previewView)
        capturedImage   = findViewById(R.id.capturedImage)
        resultDigit     = findViewById(R.id.resultDigit)
        confidenceText  = findViewById(R.id.confidenceText)
        captureBtn      = findViewById(R.id.captureBtn)
        retryBtn        = findViewById(R.id.retryBtn)
        wrongBtn        = findViewById(R.id.wrongBtn)

        cameraExecutor = Executors.newSingleThreadExecutor()
        confidenceText.text = "Загрузка модели..."

        lifecycleScope.launch {
            try {
                val loaded = withContext(Dispatchers.IO) {
                    DigitClassifier(applicationContext).also {
                        try {
                            it.restoreWeightsIfExists(checkpointFile.absolutePath)
                        } catch (_: Throwable) {
                        }
                    }
                }
                classifier = loaded
                confidenceText.text = "Наведите на цифру"
            } catch (e: Throwable) {
                modelInitError = e.message ?: e.javaClass.simpleName
                confidenceText.text = "Ошибка модели: $modelInitError"
            }
        }

        captureBtn.setOnClickListener { takePhoto() }
        retryBtn.setOnClickListener { resetToCamera() }
        wrongBtn.setOnClickListener { showCorrectionDialog() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                                         preview, imageCapture)
            } catch (e: Throwable) {
                Toast.makeText(this, "Ошибка камеры: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val previewBmp = previewView.bitmap
        if (previewBmp != null) {
            handleCaptured(scaleDownIfNeeded(previewBmp, 640))
            return
        }

        val cap = imageCapture ?: return
        captureBtn.isEnabled = false
        cap.takePicture(ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    captureBtn.isEnabled = true
                    val bmp = imageProxyToBitmap(image)
                    image.close()
                    if (bmp != null) handleCaptured(bmp)
                    else Toast.makeText(this@MainActivity,
                        "Ошибка кадра", Toast.LENGTH_SHORT).show()
                }
                override fun onError(e: ImageCaptureException) {
                    captureBtn.isEnabled = true
                    Toast.makeText(this@MainActivity,
                        "Ошибка: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun scaleDownIfNeeded(bmp: Bitmap, maxSide: Int): Bitmap {
        val w = bmp.width
        val h = bmp.height
        val largest = maxOf(w, h)
        if (largest <= maxSide) return bmp
        val scale = maxSide.toFloat() / largest
        return Bitmap.createScaledBitmap(
            bmp,
            (w * scale).toInt().coerceAtLeast(1),
            (h * scale).toInt().coerceAtLeast(1),
            true
        )
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining()).also { buffer.get(it) }

            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
            var sampleSize = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / sampleSize > 800) {
                sampleSize *= 2
            }

            val opts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts) ?: return null
            val rotate = image.imageInfo.rotationDegrees.toFloat()
            if (rotate == 0f) bmp
            else Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height,
                Matrix().also { it.postRotate(rotate) }, true)
        } catch (e: Throwable) {
            null
        }
    }

    private fun handleCaptured(bitmap: Bitmap) {
        capturedImage.setImageBitmap(bitmap)
        capturedImage.visibility = View.VISIBLE
        previewView.visibility = View.GONE
        captureBtn.visibility = View.GONE
        retryBtn.visibility = View.VISIBLE

        val activeClassifier = classifier
        if (activeClassifier == null) {
            resultDigit.text = "!"
            confidenceText.text = modelInitError?.let { "Модель не загружена: $it" } ?: "Модель ещё загружается..."
            wrongBtn.visibility = View.GONE
            return
        }

        lifecycleScope.launch {
            try {
                val input = withContext(Dispatchers.Default) {
                    ImagePreprocessor.preprocess(bitmap)
                }
                if (input == null) {
                    resultDigit.text = "?"
                    confidenceText.text = "цифра не найдена (наведите ближе / сделайте темнее)"
                    wrongBtn.visibility = View.GONE
                    return@launch
                }
                currentInput = input

                val probs = withContext(Dispatchers.Default) { activeClassifier.predict(input) }
                val digit = probs.indices.maxByOrNull { probs[it] } ?: 0
                currentDigit = digit
                resultDigit.text = digit.toString()
                confidenceText.text = "уверенность %.1f%%"
                    .format(probs[digit] * 100)
                wrongBtn.visibility = View.VISIBLE
            } catch (e: Throwable) {
                resultDigit.text = "!"
                confidenceText.text = "Ошибка: ${e.message ?: e.javaClass.simpleName}"
                wrongBtn.visibility = View.GONE
            }
        }
    }

    private fun resetToCamera() {
        capturedImage.visibility = View.GONE
        previewView.visibility = View.VISIBLE
        captureBtn.visibility = View.VISIBLE
        retryBtn.visibility = View.GONE
        resultDigit.text = "?"
        confidenceText.text = ""
        wrongBtn.visibility = View.GONE
        currentInput = null
        currentDigit = -1
    }

    private fun showCorrectionDialog() {
        val input = currentInput ?: return
        val activeClassifier = classifier ?: return
        val labels = (0..9).map { it.toString() }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Какая цифра на самом деле?")
            .setItems(labels) { _, which ->
                lifecycleScope.launch {
                    try {
                        val loss = withContext(Dispatchers.Default) {
                            activeClassifier.train(input, which)
                        }
                        withContext(Dispatchers.IO) {
                            activeClassifier.saveWeights(checkpointFile.absolutePath)
                        }
                        Toast.makeText(this@MainActivity,
                            "Дообучено (loss=${"%.3f".format(loss)})",
                            Toast.LENGTH_SHORT).show()

                        val probs = withContext(Dispatchers.Default) {
                            activeClassifier.predict(input)
                        }
                        val newDigit = probs.indices.maxByOrNull { probs[it] } ?: 0
                        resultDigit.text = newDigit.toString()
                        confidenceText.text = "после дообучения: %.1f%%"
                            .format(probs[newDigit] * 100)
                    } catch (e: Throwable) {
                        Toast.makeText(this@MainActivity,
                            "Ошибка дообучения: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier?.close()
    }
}
