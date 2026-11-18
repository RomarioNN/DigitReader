package com.example.digitreader

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.RectF

object ImagePreprocessor {

    /** Возвращает FloatArray(784) в стиле MNIST (белая цифра на чёрном) или null, если цифра не найдена */
    fun preprocess(bitmap: Bitmap): FloatArray? {
        val w = bitmap.width
        val h = bitmap.height
        if (w < 20 || h < 20) return null

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        // 1. Grayscale и bbox тёмных пикселей
        var minX = w; var minY = h; var maxX = -1; var maxY = -1; var cnt = 0
        val gray = FloatArray(w * h)
        for (i in pixels.indices) {
            val p = pixels[i]
            val g = 0.299f * (p shr 16 and 0xFF) +
                    0.587f * (p shr 8 and 0xFF) +
                    0.114f * (p and 0xFF)
            gray[i] = g
            if (g < 100f) {
                val x = i % w
                val y = i / w
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                cnt++
            }
        }
        if (cnt < 50 || maxX < 0) return null

        // 2. Crop с отступом 10%
        val pad = (maxOf(maxX - minX, maxY - minY) * 0.10f).toInt().coerceAtLeast(1)
        val x0 = (minX - pad).coerceAtLeast(0)
        val y0 = (minY - pad).coerceAtLeast(0)
        val x1 = (maxX + pad).coerceAtMost(w - 1)
        val y1 = (maxY + pad).coerceAtMost(h - 1)
        val cw = x1 - x0 + 1
        val ch = y1 - y0 + 1
        val cropped = Bitmap.createBitmap(bitmap, x0, y0, cw, ch)

        // 3. Инверсия (белая цифра на чёрном)
        val inv = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
        val invCanvas = Canvas(inv)
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(ColorMatrix(floatArrayOf(
                -1f, 0f, 0f, 0f, 255f,
                 0f,-1f, 0f, 0f, 255f,
                 0f, 0f,-1f, 0f, 255f,
                 0f, 0f, 0f, 1f,   0f
            )))
        }
        invCanvas.drawBitmap(cropped, 0f, 0f, paint)

        // 4. Ресайз в 28x28 с сохранением пропорций на чёрном фоне
        val S = 28
        val out = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
        val outCanvas = Canvas(out).apply { drawColor(Color.BLACK) }
        val scale = minOf(S.toFloat() / cw, S.toFloat() / ch)
        val nw = (cw * scale).toInt().coerceAtLeast(1)
        val nh = (ch * scale).toInt().coerceAtLeast(1)
        val dx = (S - nw) / 2f
        val dy = (S - nh) / 2f
        outCanvas.drawBitmap(inv, null,
            RectF(dx, dy, dx + nw, dy + nh),
            Paint(Paint.FILTER_BITMAP_FLAG))

        // 5. FloatArray(784) в [0,1]
        val buf = IntArray(S * S)
        out.getPixels(buf, 0, S, 0, 0, S, S)
        return FloatArray(S * S) { (buf[it] and 0xFF) / 255f }
    }
}
