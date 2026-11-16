package com.example.da3camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.sqrt

data class DetectedObject(
    val boundingBox: Rect,
    val distanceMeters: Float,
    val heightMeters: Float,
    val pixelCount: Int
)

object ObjectDetector {

    /**
     * Performs object detection strictly within the point cloud cropped to the bike lane ROI
     * at original image resolution, removes the ground plane, runs connected components,
     * filters out trees/overhead structures by height, and projects/draws bounding boxes.
     */
    fun detectObjectsFullRes(
        resizedMetricDepth: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        intrinsics: CameraIntrinsics,
        plane: PlaneResult,
        fullResMaskBytes: ByteArray,
        groundDistThresh: Float = 0.15f,
        minBlobArea: Int = 100,
        maxHeightMeters: Float = 2.5f,
        minHeightMeters: Float = 0.2f
    ): List<DetectedObject> {
        val obstacleMask = Mat.zeros(originalHeight, originalWidth, CvType.CV_8UC1)
        val maskData = ByteArray(originalWidth * originalHeight)

        for (v in 0 until originalHeight) {
            for (u in 0 until originalWidth) {
                val idx = v * originalWidth + u
                val roiVal = if (idx < fullResMaskBytes.size) fullResMaskBytes[idx].toInt() and 0xFF else 0
                if (roiVal > 0 && idx < resizedMetricDepth.size) {
                    val d = resizedMetricDepth[idx]
                    val p = RansacUtils.backprojectPoint(u.toFloat(), v.toFloat(), d, intrinsics)
                    if (p != null && p.z in 0.5f..15.0f) {
                        val distToPlane = abs(plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d)
                        if (distToPlane > groundDistThresh) {
                            maskData[idx] = 255.toByte()
                        }
                    }
                }
            }
        }
        obstacleMask.put(0, 0, maskData)

        // Morphological filtering to clean up noise and consolidate valid obstacles
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(obstacleMask, obstacleMask, Imgproc.MORPH_OPEN, kernel)
        Imgproc.morphologyEx(obstacleMask, obstacleMask, Imgproc.MORPH_CLOSE, kernel)

        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val numLabels = Imgproc.connectedComponentsWithStats(obstacleMask, labels, stats, centroids)

        val detectedObjects = mutableListOf<DetectedObject>()
        val labelsData = IntArray(originalWidth * originalHeight)
        labels.get(0, 0, labelsData)

        for (i in 1 until numLabels) {
            val left = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val top = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()

            if (area < minBlobArea) continue

            // Collect 3D points belonging to this component
            val componentPoints = mutableListOf<RansacUtils.Point3D>()
            val step = maxOf(1, (w * h) / 400)
            for (v in top until (top + h) step maxOf(1, step / 10)) {
                for (u in left until (left + w) step maxOf(1, step / 10)) {
                    if (u in 0 until originalWidth && v in 0 until originalHeight) {
                        val idx = v * originalWidth + u
                        if (labelsData[idx] == i && idx < resizedMetricDepth.size) {
                            val d = resizedMetricDepth[idx]
                            val p = RansacUtils.backprojectPoint(u.toFloat(), v.toFloat(), d, intrinsics)
                            if (p != null) {
                                componentPoints.add(p)
                            }
                        }
                    }
                }
            }

            if (componentPoints.isEmpty()) continue

            // Distance to camera (median distance of points in component)
            val distances = componentPoints.map { sqrt(it.x * it.x + it.y * it.y + it.z * it.z) }.sorted()
            val medianDist = distances[distances.size / 2]

            // Height of object (max Y - min Y among 3D points)
            val ys = componentPoints.map { it.y }
            val objHeight = abs((ys.maxOrNull() ?: 0f) - (ys.minOrNull() ?: 0f))

            // Filter out trees, overhead foliage, or structures that are too tall (> 2.5m) or too flat (< 0.2m)
            if (objHeight !in minHeightMeters..maxHeightMeters) {
                continue
            }

            detectedObjects.add(
                DetectedObject(
                    boundingBox = Rect(left, top, left + w, top + h),
                    distanceMeters = medianDist,
                    heightMeters = objHeight,
                    pixelCount = area
                )
            )
        }

        obstacleMask.release()
        labels.release()
        stats.release()
        centroids.release()
        kernel.release()

        return detectedObjects
    }

    fun drawObjectsOverlay(
        bitmap: Bitmap,
        objects: List<DetectedObject>
    ): Bitmap {
        val resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)

        val paintRect = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 6f
            isAntiAlias = true
        }

        val paintTextBg = Paint().apply {
            color = Color.argb(220, 0, 0, 0)
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        val paintText = Paint().apply {
            color = Color.WHITE
            textSize = 32f
            isAntiAlias = true
            isFakeBoldText = true
        }

        for (obj in objects) {
            val rect = obj.boundingBox
            canvas.drawRect(rect.left.toFloat(), rect.top.toFloat(), rect.right.toFloat(), rect.bottom.toFloat(), paintRect)

            val infoText = "Dist: %.1fm | H: %.1fm".format(obj.distanceMeters, obj.heightMeters)
            val textX = rect.left.toFloat().coerceAtLeast(10f)
            val textY = (rect.top - 12f).coerceAtLeast(40f)

            // Draw background rectangle for text readability
            canvas.drawRect(textX - 6f, textY - 32f, textX + 260f, textY + 10f, paintTextBg)
            canvas.drawText(infoText, textX, textY, paintText)
        }

        return resultBitmap
    }
}
