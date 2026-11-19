package com.example.digitreader

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF

data class PreprocessResult(
    val input: FloatArray,
    val preview28x28: Bitmap,
    val statsText: String
)

object ImagePreprocessor {

    /**
     * Преобразует Bitmap (с камеры, галереи или холста) в тензор 28x28 в стиле MNIST
     * (белая цифра на чёрном фоне) + возвращает Bitmap 28x28 для показа на экране и статистику.
     */
    fun preprocess(bitmap: Bitmap): PreprocessResult? {
        val w = bitmap.width
        val h = bitmap.height
        if (w < 20 || h < 20) return null

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val gray = FloatArray(w * h)
        var sumG = 0.0
        var minG = 255f
        for (i in pixels.indices) {
            val p = pixels[i]
            val g = 0.299f * (p shr 16 and 0xFF) +
                    0.587f * (p shr 8 and 0xFF) +
                    0.114f * (p and 0xFF)
            gray[i] = g
            sumG += g
            if (g < minG) minG = g
        }
        val meanG = (sumG / pixels.size).toFloat()
        if (meanG - minG < 22f) return null

        // Адаптивный порог (работает и с карандашом на сероватом листе, и с чёрным маркером)
        val threshold = (meanG - (meanG - minG) * 0.42f).coerceIn(55f, 175f)

        // 1. Поиск BBox тёмных пикселей (с небольшим игнорированием самых краёв 3%, если это фото)
        var minX = w; var minY = h; var maxX = -1; var maxY = -1; var cnt = 0
        for (i in gray.indices) {
            if (gray[i] < threshold) {
                val x = i % w
                val y = i / w
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                cnt++
            }
        }
        if (cnt < 35 || maxX < 0) return null

        // 2. Crop с отступом 10%
        val pad = (maxOf(maxX - minX, maxY - minY) * 0.10f).toInt().coerceAtLeast(2)
        val x0 = (minX - pad).coerceAtLeast(0)
        val y0 = (minY - pad).coerceAtLeast(0)
        val x1 = (maxX + pad).coerceAtMost(w - 1)
        val y1 = (maxY + pad).coerceAtMost(h - 1)
        val cw = x1 - x0 + 1
        val ch = y1 - y0 + 1

        // 3. Инверсия с контрастной нормализацией (фон -> 0, штрихи -> до 255)
        val invPixels = IntArray(cw * ch)
        val denom = (threshold - minG).coerceAtLeast(15f)
        for (y in 0 until ch) {
            val srcRow = (y0 + y) * w + x0
            val dstRow = y * cw
            for (x in 0 until cw) {
                val g = gray[srcRow + x]
                val v = if (g >= threshold) {
                    0
                } else {
                    (((threshold - g) / denom) * 255f).toInt().coerceIn(0, 255)
                }
                invPixels[dstRow + x] = Color.rgb(v, v, v)
            }
        }
        val inv = Bitmap.createBitmap(invPixels, cw, ch, Bitmap.Config.ARGB_8888)

        // 4. Ресайз в 28x28 (вписываем в область 22x22 с полями как в MNIST)
        val size = 28
        val inner = 22f
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val outCanvas = Canvas(out).apply { drawColor(Color.BLACK) }
        val scale = minOf(inner / cw, inner / ch)
        val nw = (cw * scale).coerceAtLeast(1f)
        val nh = (ch * scale).coerceAtLeast(1f)
        val dx = (size - nw) / 2f
        val dy = (size - nh) / 2f
        outCanvas.drawBitmap(
            inv,
            null,
            RectF(dx, dy, dx + nw, dy + nh),
            Paint(Paint.FILTER_BITMAP_FLAG)
        )

        // 5. FloatArray(784) в [0, 1]
        val buf = IntArray(size * size)
        out.getPixels(buf, 0, size, 0, 0, size, size)
        var active = 0
        val input = FloatArray(size * size) { i ->
            val f = (buf[i] and 0xFF) / 255f
            if (f > 0.1f) active++
            f
        }

        val stats = "BBox ${cw}x${ch}px | порог ${threshold.toInt()} | акт. $active/784"
        return PreprocessResult(input, out, stats)
    }
}
