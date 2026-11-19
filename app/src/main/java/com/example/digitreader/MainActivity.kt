package com.example.digitreader

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
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
    private lateinit var drawingView: DrawingView
    private lateinit var capturedImage: ImageView
    private lateinit var mnistPreviewImage: ImageView
    private lateinit var preprocessStatsText: TextView
    private lateinit var resultDigit: TextView
    private lateinit var confidenceText: TextView
    private lateinit var ckptStatusBadge: TextView
    private lateinit var trainingLogHeader: TextView
    private lateinit var trainingLogText: TextView
    private lateinit var probsBarsContainer: LinearLayout

    private lateinit var modeCameraBtn: Button
    private lateinit var modeSheetBtn: Button
    private lateinit var loadPhotoBtn: Button
    private lateinit var captureBtn: Button
    private lateinit var clearSheetBtn: Button
    private lateinit var retryBtn: Button
    private lateinit var wrongBtn: Button
    private lateinit var resetWeightsBtn: Button

    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    private var classifier: DigitClassifier? = null
    private var modelInitError: String? = null
    private var currentInput: FloatArray? = null
    private var currentDigit: Int = -1
    private var isSheetMode: Boolean = false
    private var trainStepsCount: Int = 0
    private val trainingLogs = mutableListOf<String>()

    private val checkpointFile: File by lazy {
        File(filesDir, "digit_weights.ckpt")
    }

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Нужна камера (или используйте режим Лист / Фото)", Toast.LENGTH_LONG).show()
        }

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                loadBitmapFromUri(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView         = findViewById(R.id.previewView)
        drawingView         = findViewById(R.id.drawingView)
        capturedImage       = findViewById(R.id.capturedImage)
        mnistPreviewImage   = findViewById(R.id.mnistPreviewImage)
        preprocessStatsText = findViewById(R.id.preprocessStatsText)
        resultDigit         = findViewById(R.id.resultDigit)
        confidenceText      = findViewById(R.id.confidenceText)
        ckptStatusBadge     = findViewById(R.id.ckptStatusBadge)
        trainingLogHeader   = findViewById(R.id.trainingLogHeader)
        trainingLogText     = findViewById(R.id.trainingLogText)
        probsBarsContainer  = findViewById(R.id.probsBarsContainer)

        modeCameraBtn       = findViewById(R.id.modeCameraBtn)
        modeSheetBtn        = findViewById(R.id.modeSheetBtn)
        loadPhotoBtn        = findViewById(R.id.loadPhotoBtn)
        captureBtn          = findViewById(R.id.captureBtn)
        clearSheetBtn       = findViewById(R.id.clearSheetBtn)
        retryBtn            = findViewById(R.id.retryBtn)
        wrongBtn            = findViewById(R.id.wrongBtn)
        resetWeightsBtn     = findViewById(R.id.resetWeightsBtn)

        cameraExecutor = Executors.newSingleThreadExecutor()
        renderProbsBars(null, -1)
        loadModelAsync(restoreCkpt = true)

        modeCameraBtn.setOnClickListener { switchMode(sheet = false) }
        modeSheetBtn.setOnClickListener { switchMode(sheet = true) }
        loadPhotoBtn.setOnClickListener { pickImageLauncher.launch("image/*") }
        captureBtn.setOnClickListener { onCaptureClicked() }
        clearSheetBtn.setOnClickListener {
            drawingView.clear()
            resetResultOnly()
        }
        retryBtn.setOnClickListener { resetToInputMode() }
        wrongBtn.setOnClickListener { showCorrectionDialog() }
        resetWeightsBtn.setOnClickListener { resetModelWeights() }

        updateModeButtonsUI()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun loadModelAsync(restoreCkpt: Boolean) {
        confidenceText.text = "Загрузка модели..."
        lifecycleScope.launch {
            try {
                var restored = false
                val loaded = withContext(Dispatchers.IO) {
                    DigitClassifier(applicationContext).also {
                        if (restoreCkpt && checkpointFile.exists()) {
                            try {
                                it.restoreWeightsIfExists(checkpointFile.absolutePath)
                                restored = true
                            } catch (_: Throwable) {
                            }
                        }
                    }
                }
                classifier?.close()
                classifier = loaded
                modelInitError = null
                updateCkptBadge(restored || trainStepsCount > 0)
                confidenceText.text = if (isSheetMode) "Нарисуйте цифру" else "Наведите на цифру"
            } catch (e: Throwable) {
                modelInitError = e.message ?: e.javaClass.simpleName
                confidenceText.text = "Ошибка модели: $modelInitError"
            }
        }
    }

    private fun resetModelWeights() {
        if (checkpointFile.exists()) {
            checkpointFile.delete()
        }
        trainStepsCount = 0
        trainingLogs.clear()
        trainingLogHeader.text = "Журнал дообучения (шагов: 0)"
        trainingLogText.text = "Веса сброшены к заводским MNIST."
        loadModelAsync(restoreCkpt = false)
        Toast.makeText(this, "Веса сброшены к базовым", Toast.LENGTH_SHORT).show()
    }

    private fun updateCkptBadge(isCustom: Boolean) {
        if (isCustom) {
            ckptStatusBadge.text = if (trainStepsCount > 0) "ckpt: дообучен ($trainStepsCount)" else "ckpt: сохранён"
            ckptStatusBadge.setTextColor(Color.parseColor("#6EE7B7"))
        } else {
            ckptStatusBadge.text = "ckpt: заводской"
            ckptStatusBadge.setTextColor(Color.parseColor("#D1D5DB"))
        }
    }

    private fun switchMode(sheet: Boolean) {
        isSheetMode = sheet
        resetToInputMode()
        updateModeButtonsUI()
    }

    private fun updateModeButtonsUI() {
        modeCameraBtn.alpha = if (!isSheetMode) 1.0f else 0.55f
        modeSheetBtn.alpha = if (isSheetMode) 1.0f else 0.55f
        captureBtn.text = if (isSheetMode) "Распознать" else "Сделать фото"
        clearSheetBtn.visibility = if (isSheetMode && capturedImage.visibility != View.VISIBLE) View.VISIBLE else View.GONE
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
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Throwable) {
                Toast.makeText(this, "Ошибка камеры: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onCaptureClicked() {
        if (isSheetMode) {
            if (drawingView.isEmpty()) {
                Toast.makeText(this, "Сначала нарисуйте цифру на листе", Toast.LENGTH_SHORT).show()
                return
            }
            handleCaptured(drawingView.exportBitmap())
            return
        }

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
                    else Toast.makeText(this@MainActivity, "Ошибка кадра", Toast.LENGTH_SHORT).show()
                }
                override fun onError(e: ImageCaptureException) {
                    captureBtn.isEnabled = true
                    Toast.makeText(this@MainActivity, "Ошибка: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun loadBitmapFromUri(uri: Uri) {
        lifecycleScope.launch {
            val bmp = withContext(Dispatchers.IO) {
                try {
                    contentResolver.openInputStream(uri)?.use { stream ->
                        val bytes = stream.readBytes()
                        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                        var sample = 1
                        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 800) {
                            sample *= 2
                        }
                        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
                        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
                    }
                } catch (_: Throwable) {
                    null
                }
            }
            if (bmp != null) {
                handleCaptured(bmp)
            } else {
                Toast.makeText(this@MainActivity, "Не удалось открыть фото", Toast.LENGTH_SHORT).show()
            }
        }
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
        } catch (_: Throwable) {
            null
        }
    }

    private fun handleCaptured(bitmap: Bitmap) {
        capturedImage.setImageBitmap(bitmap)
        capturedImage.visibility = View.VISIBLE
        previewView.visibility = View.GONE
        drawingView.visibility = View.GONE
        captureBtn.visibility = View.GONE
        clearSheetBtn.visibility = View.GONE
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
                val prep = withContext(Dispatchers.Default) {
                    ImagePreprocessor.preprocess(bitmap)
                }
                if (prep == null) {
                    resultDigit.text = "?"
                    confidenceText.text = "цифра не найдена (сделайте контрастнее)"
                    preprocessStatsText.text = "контур не найден"
                    mnistPreviewImage.setImageDrawable(null)
                    renderProbsBars(null, -1)
                    wrongBtn.visibility = View.GONE
                    return@launch
                }

                // Отображаем обработанное 28x28 изображение без сглаживания (пиксель в пиксель)
                val drawable = BitmapDrawable(resources, prep.preview28x28).apply {
                    isFilterBitmap = false
                }
                mnistPreviewImage.setImageDrawable(drawable)
                preprocessStatsText.text = prep.statsText

                currentInput = prep.input
                val probs = withContext(Dispatchers.Default) { activeClassifier.predict(prep.input) }
                val digit = probs.indices.maxByOrNull { probs[it] } ?: 0
                currentDigit = digit
                resultDigit.text = digit.toString()
                confidenceText.text = "уверенность %.1f%%".format(probs[digit] * 100)
                renderProbsBars(probs, digit)
                wrongBtn.visibility = View.VISIBLE
            } catch (e: Throwable) {
                resultDigit.text = "!"
                confidenceText.text = "Ошибка: ${e.message ?: e.javaClass.simpleName}"
                wrongBtn.visibility = View.GONE
            }
        }
    }

    private fun renderProbsBars(probs: FloatArray?, bestDigit: Int) {
        probsBarsContainer.removeAllViews()
        val density = resources.displayMetrics.density
        val maxBarHeightPx = (54 * density).toInt()

        for (d in 0..9) {
            val p = if (probs != null && d < probs.size) probs[d] * 100f else 0f
            val isBest = d == bestDigit

            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f)
                setPadding(2, 0, 2, 0)
            }

            val pctText = TextView(this).apply {
                text = if (p >= 1f) "${p.toInt()}%" else ""
                textSize = 9f
                setTextColor(if (isBest) Color.parseColor("#6EE7B7") else Color.parseColor("#9CA3AF"))
                gravity = Gravity.CENTER
            }

            val barHeight = ((p / 100f) * maxBarHeightPx).toInt().coerceAtLeast((4 * density).toInt())
            val bar = View(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    barHeight
                ).apply {
                    setMargins(3, 2, 3, 2)
                }
                setBackgroundColor(
                    if (isBest) Color.parseColor("#44DD44")
                    else Color.parseColor("#4F46E5")
                )
            }

            val label = TextView(this).apply {
                text = d.toString()
                textSize = 11f
                setTextColor(if (isBest) Color.WHITE else Color.parseColor("#9CA3AF"))
                gravity = Gravity.CENTER
            }

            col.addView(pctText)
            col.addView(bar)
            col.addView(label)
            probsBarsContainer.addView(col)
        }
    }

    private fun resetResultOnly() {
        resultDigit.text = "?"
        confidenceText.text = if (isSheetMode) "Нарисуйте цифру" else "Наведите на цифру"
        preprocessStatsText.text = "ожидание кадра"
        mnistPreviewImage.setImageDrawable(null)
        renderProbsBars(null, -1)
        wrongBtn.visibility = View.GONE
        currentInput = null
        currentDigit = -1
    }

    private fun resetToInputMode() {
        capturedImage.visibility = View.GONE
        previewView.visibility = if (isSheetMode) View.GONE else View.VISIBLE
        drawingView.visibility = if (isSheetMode) View.VISIBLE else View.GONE
        captureBtn.visibility = View.VISIBLE
        clearSheetBtn.visibility = if (isSheetMode) View.VISIBLE else View.GONE
        retryBtn.visibility = View.GONE
        resetResultOnly()
    }

    private fun showCorrectionDialog() {
        val input = currentInput ?: return
        val activeClassifier = classifier ?: return
        val prevDigit = currentDigit
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
                        trainStepsCount++
                        updateCkptBadge(true)

                        Toast.makeText(
                            this@MainActivity,
                            "Дообучено (loss=${"%.3f".format(loss)})",
                            Toast.LENGTH_SHORT
                        ).show()

                        val probs = withContext(Dispatchers.Default) {
                            activeClassifier.predict(input)
                        }
                        val newDigit = probs.indices.maxByOrNull { probs[it] } ?: 0
                        currentDigit = newDigit
                        resultDigit.text = newDigit.toString()
                        confidenceText.text = "после дообучения: %.1f%%".format(probs[newDigit] * 100)
                        renderProbsBars(probs, newDigit)

                        val logLine = "#$trainStepsCount: было $prevDigit → цель $which | loss=${"%.3f".format(loss)} | ответ $newDigit (${"%.1f".format(probs[newDigit] * 100)}%)"
                        trainingLogs.add(0, logLine)
                        trainingLogHeader.text = "Журнал дообучения (шагов: $trainStepsCount)"
                        trainingLogText.text = trainingLogs.take(12).joinToString("
")
                    } catch (e: Throwable) {
                        Toast.makeText(
                            this@MainActivity,
                            "Ошибка дообучения: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
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
