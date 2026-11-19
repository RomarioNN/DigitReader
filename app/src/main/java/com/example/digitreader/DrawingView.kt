package com.example.digitreader

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class DrawingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val drawPath = Path()
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#141414")
        style = Paint.Style.STROKE
        strokeWidth = 34f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCCCCC")
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private var hasStroke = false

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.parseColor("#F5F5F5"))
        canvas.drawRect(18f, 18f, width - 18f, height - 18f, borderPaint)
        canvas.drawPath(drawPath, strokePaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                drawPath.moveTo(x, y)
                drawPath.lineTo(x + 0.5f, y + 0.5f)
                hasStroke = true
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                drawPath.lineTo(x, y)
                hasStroke = true
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    fun clear() {
        drawPath.reset()
        hasStroke = false
        invalidate()
    }

    fun isEmpty(): Boolean = !hasStroke

    fun exportBitmap(): Bitmap {
        val w = width.coerceAtLeast(240)
        val h = height.coerceAtLeast(240)
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.parseColor("#F5F5F5"))
        c.drawPath(drawPath, strokePaint)
        return bmp
    }
}
