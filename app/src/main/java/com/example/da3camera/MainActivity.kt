
package com.example.da3camera

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import android.widget.EditText
import org.opencv.android.OpenCVLoader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import android.view.View
import android.content.Intent
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import android.graphics.BitmapFactory
import java.io.File
import androidx.exifinterface.media.ExifInterface
import android.graphics.Matrix
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import java.util.concurrent.CompletableFuture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager


class MainActivity : AppCompatActivity() {

    private var capturedCameraIntrinsics: CameraIntrinsics? = null

    private var da3Module: org.pytorch.executorch.Module? = null
    private var selectedBitmap: Bitmap? = null

    private var selectedImageFileName: String? = null

    private lateinit var imagePreview: ImageView
    private lateinit var textImageInfo: TextView

    private lateinit var depthPreview: ImageView
    private lateinit var textDepthTitle: TextView

    private lateinit var textOutputFolder: TextView

    private var outputFolderUri: Uri? = null

    private val requestCameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->

            if (isGranted) {
                Log.d("DA3_CAMERA", "Camera permission granted by user")

                cameraLauncher.launch(
                    Intent(this, CameraActivity::class.java)
                )

            } else {
                Log.d("DA3_CAMERA", "Camera permission denied")

                Toast.makeText(
                    this,
                    "Camera permission is required to take photos",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    private fun loadCorrectlyOrientedBitmap(photoPath: String): Bitmap? {

        val bitmap = BitmapFactory.decodeFile(photoPath) ?: return null

        val exif = ExifInterface(photoPath)

        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        Log.d("DA3_CAMERA", "JPEG EXIF orientation: $orientation")

        val matrix = Matrix()

        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)

            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }
        }

        if (orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED
        ) {
            return bitmap
        }

        val rotatedBitmap = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )

        if (rotatedBitmap !== bitmap) {
            bitmap.recycle()
        }

        return rotatedBitmap
    }

    private fun createExecuTorchInput(input: FloatArray): EValue {

        val expectedElements = 1 * 3 * 504 * 504

        require(input.size == expectedElements) {
            "Invalid input size: ${input.size}. Expected $expectedElements"
        }

        val tensor = Tensor.fromBlob(
            input,
            longArrayOf(1, 3, 504, 504)
        )

        android.util.Log.d(
            "DA3_INFERENCE",
            "ExecuTorch input tensor created successfully"
        )

        android.util.Log.d(
            "DA3_INFERENCE",
            "Input shape: [1, 3, 504, 504]"
        )

        return EValue.from(tensor)
    }

    private fun saveClassificationTxt(
        predictedClassName: String,
        inputFileName: String
    ) {
        val folderUri = outputFolderUri

        if (folderUri == null) {
            runOnUiThread {
                Toast.makeText(
                    this,
                    "Please select an output folder first",
                    Toast.LENGTH_LONG
                ).show()
            }
            return
        }

        try {
            // Preserve the original image filename without its extension.
            val baseName = inputFileName.substringBeforeLast(
                ".",
                inputFileName
            )

            val outputFileName = "${baseName}_classification.txt"

            // Encode the predicted class name as UTF-8 text.
            val txtBytes = predictedClassName.toByteArray(Charsets.UTF_8)

            // Access the same folder used for depth and segmentation.
            val folder = DocumentFile.fromTreeUri(this, folderUri)
                ?: error("Cannot access selected output folder")

            // Replace an existing classification file.
            folder.findFile(outputFileName)?.let { existingFile ->
                check(existingFile.delete()) {
                    "Cannot replace existing classification TXT"
                }
            }

            // Create the text document.
            val outputFile = folder.createFile(
                "text/plain",
                outputFileName
            ) ?: error("Cannot create classification TXT")

            // Write the predicted class name.
            contentResolver.openOutputStream(outputFile.uri, "w")?.use {
                it.write(txtBytes)
            } ?: error("Cannot open classification TXT output stream")

            // Read back the saved file.
            val savedBytes = contentResolver.openInputStream(outputFile.uri)?.use {
                it.readBytes()
            } ?: error("Cannot read saved classification TXT")

            // Verify the saved contents.
            check(savedBytes.contentEquals(txtBytes)) {
                "Saved classification TXT differs from original text"
            }

            Log.d(
                "DA3_CLASS",
                "Saved classification TXT verified successfully: " +
                        "${savedBytes.size} bytes, byte-for-byte identical"
            )

            Log.d(
                "DA3_CLASS",
                "Saved classification TXT: $outputFileName"
            )

            Log.d(
                "DA3_CLASS",
                "Saved classification content: $predictedClassName"
            )

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Saved: $outputFileName",
                    Toast.LENGTH_LONG
                ).show()
            }

        } catch (e: Exception) {
            Log.e(
                "DA3_CLASS",
                "Failed to save classification TXT",
                e
            )

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Classification TXT save failed: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun saveSegmentationMaskPng(
        pngBytes: ByteArray,
        inputFileName: String
    ) {
        val folderUri = outputFolderUri

        if (folderUri == null) {
            runOnUiThread {
                Toast.makeText(
                    this,
                    "Please select an output folder first",
                    Toast.LENGTH_LONG
                ).show()
            }
            return
        }

        try {
            // Remove the original image extension.
            val baseName = inputFileName.substringBeforeLast(
                ".",
                inputFileName
            )

            // Generate the segmentation output filename.
            val outputFileName = "${baseName}_mask.png"

            // Access the user-selected output folder.
            val folder = DocumentFile.fromTreeUri(this, folderUri)
                ?: error("Cannot access selected output folder")

            // Replace an existing mask with the same filename.
            folder.findFile(outputFileName)?.let { existingFile ->
                check(existingFile.delete()) {
                    "Cannot replace existing segmentation PNG"
                }
            }

            // Create the PNG document.
            val outputFile = folder.createFile(
                "image/png",
                outputFileName
            ) ?: error("Cannot create segmentation PNG")

            // Write the PNG bytes.
            contentResolver.openOutputStream(outputFile.uri, "w")?.use {
                it.write(pngBytes)
            } ?: error("Cannot open segmentation PNG output stream")

            // Read back the saved file.
            val savedBytes = contentResolver.openInputStream(outputFile.uri)?.use {
                it.readBytes()
            } ?: error("Cannot read saved segmentation PNG")

            // Verify that the saved PNG is byte-for-byte identical.
            check(savedBytes.contentEquals(pngBytes)) {
                "Saved segmentation PNG differs from encoded PNG"
            }

            Log.d(
                "DA3_SEG",
                "Saved segmentation PNG verified successfully: " +
                        "${savedBytes.size} bytes, byte-for-byte identical"
            )

            Log.d(
                "DA3_SEG",
                "Saved segmentation PNG: $outputFileName " +
                        "(${pngBytes.size} bytes)"
            )

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Saved: $outputFileName",
                    Toast.LENGTH_LONG
                ).show()
            }

        } catch (e: Exception) {

            Log.e(
                "DA3_SEG",
                "Failed to save segmentation PNG",
                e
            )

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Segmentation PNG save failed: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun drawDetectedLaneLines(canvas: Canvas, laneResult: Ufldv2Utils.LaneResult) {
        val lineWidth = maxOf(2f, canvas.width / 300f)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            strokeWidth = lineWidth
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
        // Preserve the detector's lane identities, even if ROI construction swaps them.
        for ((points, color) in listOf(
            laneResult.detectedLeftLane to Color.RED,
            laneResult.detectedRightLane to Color.CYAN
        )) {
            val sorted = points.sortedBy { it.y }
            if (sorted.isEmpty()) continue
            paint.color = color
            paint.style = Paint.Style.STROKE
            val path = Path().apply {
                moveTo(sorted.first().x, sorted.first().y)
                for (point in sorted.drop(1)) lineTo(point.x, point.y)
            }
            canvas.drawPath(path, paint)
            paint.style = Paint.Style.FILL
            for (point in sorted) canvas.drawCircle(point.x, point.y, lineWidth * 1.5f, paint)
        }
    }

    private fun saveBikeLaneOverlayPng(
        originalBitmap: Bitmap,
        laneResult: Ufldv2Utils.LaneResult,
        inputFileName: String,
        detectionOnly: Boolean = false
    ) {
        val folderUri = outputFolderUri ?: return
        try {
            val width = originalBitmap.width
            val height = originalBitmap.height
            val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)

            val maskBytes = laneResult.maskBytes
            if (!detectionOnly && maskBytes.isNotEmpty()) {
                val pixels = IntArray(width * height)
                mutableBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
                val len = minOf(pixels.size, maskBytes.size)
                for (i in 0 until len) {
                    if ((maskBytes[i].toInt() and 0xFF) > 0) {
                        val color = pixels[i]
                        val r = ((color shr 16 and 0xFF) * 0.5f).toInt()
                        val g = ((color shr 8 and 0xFF) * 0.5f + 255 * 0.5f).toInt()
                        val b = ((color and 0xFF) * 0.5f + 100 * 0.5f).toInt()
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
                mutableBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            }

            drawDetectedLaneLines(canvas, laneResult)

            val stream = java.io.ByteArrayOutputStream()
            mutableBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            val pngBytes = stream.toByteArray()

            val baseName = inputFileName.substringBeforeLast('.', inputFileName)
            val outputFileName = if (detectionOnly) "${baseName}_lane_detection.png"
                else "${baseName}_bikelane_overlay.png"

            val folder = DocumentFile.fromTreeUri(this, folderUri) ?: return
            folder.findFile(outputFileName)?.delete()
            val outputFile = folder.createFile("image/png", outputFileName) ?: return
            contentResolver.openOutputStream(outputFile.uri, "w")?.use {
                it.write(pngBytes)
            }

            Log.d("UFLDV2", "Saved bike lane overlay PNG with border lines: $outputFileName")
        } catch (e: Exception) {
            Log.e("UFLDV2", "Failed to save bike lane overlay PNG", e)
        }
    }

    private fun saveMetricDepthTiff(
        tiffBytes: ByteArray,
        inputFileName: String
    ) {
        val folderUri = outputFolderUri

        if (folderUri == null) {
            runOnUiThread {
                Toast.makeText(
                    this,
                    "Please select an output folder first",
                    Toast.LENGTH_LONG
                ).show()
            }
            return
        }

        try {
            // Remove the original file extension.
            val baseName = inputFileName.substringBeforeLast(
                ".",
                inputFileName
            )

            // Generate the output filename.
            val outputFileName = "${baseName}_depth.tiff"

            // Access the folder selected by the user.
            val folder = DocumentFile.fromTreeUri(this, folderUri)
                ?: error("Cannot access selected output folder")

            // Replace an existing file with the same name.
            folder.findFile(outputFileName)?.let { existingFile ->
                check(existingFile.delete()) {
                    "Cannot replace existing TIFF file"
                }
            }

            // Create the TIFF document.
            val outputFile = folder.createFile(
                "image/tiff",
                outputFileName
            ) ?: error("Cannot create TIFF file")

            // Write the TIFF bytes.
            contentResolver.openOutputStream(outputFile.uri, "w")?.use {
                it.write(tiffBytes)
            } ?: error("Cannot open TIFF output stream")

            // Verify the saved TIFF without loading it entirely into memory.
            val buffer = ByteArray(64 * 1024)
            var verifiedBytes = 0

            contentResolver.openInputStream(outputFile.uri)?.use { input ->

                while (true) {
                    val bytesRead = input.read(buffer)

                    if (bytesRead == -1) {
                        break
                    }

                    check(verifiedBytes + bytesRead <= tiffBytes.size) {
                        "Saved TIFF is larger than the encoded TIFF"
                    }

                    for (i in 0 until bytesRead) {
                        check(buffer[i] == tiffBytes[verifiedBytes + i]) {
                            "Saved TIFF differs from the encoded TIFF " +
                                    "at byte ${verifiedBytes + i}"
                        }
                    }

                    verifiedBytes += bytesRead
                }

            } ?: error("Cannot read saved TIFF")

            check(verifiedBytes == tiffBytes.size) {
                "Saved TIFF size differs from the encoded TIFF"
            }

            Log.d(
                "DA3_TIFF",
                "Saved TIFF verified successfully: " +
                        "$verifiedBytes bytes, byte-for-byte identical"
            )

            Log.d(
                "DA3_TIFF",
                "Saved TIFF: $outputFileName (${tiffBytes.size} bytes)"
            )

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Saved: $outputFileName",
                    Toast.LENGTH_LONG
                ).show()
            }

        } catch (e: Exception) {

            Log.e(
                "DA3_TIFF",
                "Failed to save TIFF",
                e
            )

            runOnUiThread {
                Toast.makeText(
                    this,
                    "TIFF save failed: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun prepareSegmentationMask(
        segmentationLogits: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        padTop: Int,
        padBottom: Int,
        padLeft: Int,
        padRight: Int
    ): ByteArray {

        val modelSize = 504
        val numClasses = 21
        val modelPixels = modelSize * modelSize

        require(segmentationLogits.size == numClasses * modelPixels) {
            "Unexpected segmentation logits size"
        }

        require(originalWidth > 0 && originalHeight > 0) {
            "Invalid output dimensions"
        }

        val validWidth = modelSize - padLeft - padRight
        val validHeight = modelSize - padTop - padBottom

        require(validWidth > 0 && validHeight > 0) {
            "Invalid segmentation padding"
        }

        val outputPixels = originalWidth * originalHeight

        // Best class score at each output pixel.
        val bestScores = FloatArray(outputPixels) {
            Float.NEGATIVE_INFINITY
        }

        // Final segmentation class IDs: 0 through 20.
        val mask = ByteArray(outputPixels)

        val cropRect = org.opencv.core.Rect(
            padLeft,
            padTop,
            validWidth,
            validHeight
        )

        // Process one class channel at a time.
        for (classIndex in 0 until numClasses) {

            val offset = classIndex * modelPixels

            val classLogits = segmentationLogits.copyOfRange(
                offset,
                offset + modelPixels
            )

            val fullMat = Mat(
                modelSize,
                modelSize,
                CvType.CV_32FC1
            )

            try {
                fullMat.put(0, 0, classLogits)

                val croppedMat = Mat(fullMat, cropRect)
                val resizedMat = Mat()

                try {
                    Imgproc.resize(
                        croppedMat,
                        resizedMat,
                        Size(
                            originalWidth.toDouble(),
                            originalHeight.toDouble()
                        ),
                        0.0,
                        0.0,
                        Imgproc.INTER_LINEAR
                    )

                    val resizedLogits = FloatArray(outputPixels)

                    resizedMat.get(
                        0,
                        0,
                        resizedLogits
                    )

                    // Argmax across the 21 class channels.
                    for (i in 0 until outputPixels) {

                        val score = resizedLogits[i]

                        if (score > bestScores[i]) {
                            bestScores[i] = score
                            mask[i] = classIndex.toByte()
                        }
                    }

                } finally {
                    croppedMat.release()
                    resizedMat.release()
                }

            } finally {
                fullMat.release()
            }
        }

        return mask
    }

    private fun prepareMetricDepthMap(
        metricDepthValues: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        padTop: Int,
        padBottom: Int,
        padLeft: Int,
        padRight: Int
    ): FloatArray {

        require(metricDepthValues.size == 504 * 504) {
            "Unexpected metric depth array size"
        }

        // Create a Float32 OpenCV matrix from the model output.
        val fullDepthMat = Mat(504, 504, CvType.CV_32FC1)

        try {
            fullDepthMat.put(0, 0, metricDepthValues)

            // Remove the padding added during preprocessing.
            val validWidth = 504 - padLeft - padRight
            val validHeight = 504 - padTop - padBottom

            require(validWidth > 0 && validHeight > 0) {
                "Invalid depth padding"
            }

            val cropRect = org.opencv.core.Rect(
                padLeft,
                padTop,
                validWidth,
                validHeight
            )

            val croppedDepthMat = Mat(fullDepthMat, cropRect)
            val resizedDepthMat = Mat()

            try {
                // Resize to the original input image dimensions.
                org.opencv.imgproc.Imgproc.resize(
                    croppedDepthMat,
                    resizedDepthMat,
                    org.opencv.core.Size(
                        originalWidth.toDouble(),
                        originalHeight.toDouble()
                    ),
                    0.0,
                    0.0,
                    org.opencv.imgproc.Imgproc.INTER_LINEAR
                )

                // Preserve the Float32 values in meters.
                val resizedMetricDepth = FloatArray(
                    originalWidth * originalHeight
                )

                resizedDepthMat.get(
                    0,
                    0,
                    resizedMetricDepth
                )

                return resizedMetricDepth

            } finally {
                croppedDepthMat.release()
                resizedDepthMat.release()
            }

        } finally {
            fullDepthMat.release()
        }
    }

    private fun getFileNameFromUri(uri: Uri): String {

        var fileName: String? = null

        contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null
        )?.use { cursor ->

            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(
                    OpenableColumns.DISPLAY_NAME
                )

                if (nameIndex >= 0) {
                    fileName = cursor.getString(nameIndex)
                }
            }
        }

        Log.d("DA3_FILENAME", "Selected URI: $uri")
        Log.d("DA3_FILENAME", "Android DISPLAY_NAME: $fileName")

        return fileName ?: "image.png"
    }

    private fun createDepthBitmap(
        depthValues: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        padTop: Int,
        padBottom: Int,
        padLeft: Int,
        padRight: Int
    ): Bitmap {

        val modelSize = 504

        require(depthValues.size == modelSize * modelSize) {
            "Unexpected depth tensor size: ${depthValues.size}"
        }

        val validWidth = modelSize - padLeft - padRight
        val validHeight = modelSize - padTop - padBottom

        require(validWidth > 0 && validHeight > 0) {
            "Invalid depth crop dimensions"
        }

        val depthMat = Mat(modelSize, modelSize, CvType.CV_32FC1)

        try {
            depthMat.put(0, 0, depthValues)

            val croppedDepth = depthMat.submat(
                org.opencv.core.Rect(
                    padLeft,
                    padTop,
                    validWidth,
                    validHeight
                )
            )

            try {
                val resizedDepth = Mat()

                try {
                    Imgproc.resize(
                        croppedDepth,
                        resizedDepth,
                        Size(
                            originalWidth.toDouble(),
                            originalHeight.toDouble()
                        ),
                        0.0,
                        0.0,
                        Imgproc.INTER_LINEAR
                    )

                    val resizedValues = FloatArray(originalWidth * originalHeight)
                    resizedDepth.get(0, 0, resizedValues)

                    var minDepth = Float.POSITIVE_INFINITY
                    var maxDepth = Float.NEGATIVE_INFINITY

                    for (depth in resizedValues) {
                        if (depth.isFinite()) {
                            if (depth < minDepth) {
                                minDepth = depth
                            }

                            if (depth > maxDepth) {
                                maxDepth = depth
                            }
                        }
                    }

                    require(minDepth.isFinite() && maxDepth.isFinite()) {
                        "No valid depth values"
                    }

                    val range = (maxDepth - minDepth).coerceAtLeast(1e-6f)

                    val pixels = IntArray(resizedValues.size)

                    for (i in resizedValues.indices) {
                        val depth = resizedValues[i]

                        val intensity = if (depth.isFinite()) {
                            (((depth - minDepth) / range) * 255f)
                                .toInt()
                                .coerceIn(0, 255)
                        } else {
                            0
                        }

                        pixels[i] = android.graphics.Color.rgb(
                            intensity,
                            intensity,
                            intensity
                        )
                    }

                    return Bitmap.createBitmap(
                        pixels,
                        originalWidth,
                        originalHeight,
                        Bitmap.Config.ARGB_8888
                    )

                } finally {
                    resizedDepth.release()
                }

            } finally {
                croppedDepth.release()
            }

        } finally {
            depthMat.release()
        }
    }

    private fun runDA3Inference(
        input: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        padTop: Int,
        padBottom: Int,
        padLeft: Int,
        padRight: Int,
        correctedFx: Float,
        correctedFy: Float,
        correctedCx: Float,
        correctedCy: Float
    ) {

        val module = da3Module

        if (module == null) {
            Toast.makeText(
                this,
                "DA3 model is still loading. Please wait.",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        Thread {
            try {
                android.util.Log.d(
                    "DA3_INFERENCE",
                    "Starting DA3 inference..."
                )

                // Launch UFLD v2 lane detection concurrently in parallel with DA3 model inference
                val laneFuture = java.util.concurrent.CompletableFuture.supplyAsync {
                    try {
                        val bmp = selectedBitmap
                        if (bmp != null) {
                            Ufldv2Utils.generateLaneResult(bmp, originalWidth, originalHeight)
                        } else {
                            Ufldv2Utils.LaneResult(ByteArray(originalWidth * originalHeight), ArrayList(), ArrayList(), 0f, 0f)
                        }
                    } catch (e: Exception) {
                        Log.e("UFLDV2", "Failed to detect lane lines in parallel", e)
                        Ufldv2Utils.LaneResult(ByteArray(originalWidth * originalHeight), ArrayList(), ArrayList(), 0f, 0f)
                    }
                }

                val inputEValue = createExecuTorchInput(input)

                val startTime = System.currentTimeMillis()

                val outputs = module.forward(inputEValue)

                // Output 0 contains the predicted depth map
                val depthValues = outputs[0].toTensor().dataAsFloatArray

                // Extract segmentation logits from ExecuTorch.
                val segmentationLogits =
                    outputs[1].toTensor().dataAsFloatArray

                // Extract classification logits: [1, 5].
                val classificationLogits = outputs[2].toTensor().dataAsFloatArray

                // Classification labels in model output order.
                val classificationLabels = arrayOf(
                    "bidirectional bicycle lane",
                    "unidirectional bicycle lane",
                    "roadway",
                    "shared streets",
                    "sidewalk"
                )

                // Validate classification output.
                check(classificationLogits.size == classificationLabels.size) {
                    "Unexpected classification output size: ${classificationLogits.size}"
                }

                check(classificationLogits.all { it.isFinite() }) {
                    "Classification logits contain NaN or infinity"
                }

                // Select the class with the highest logit.
                val predictedClassId = classificationLogits.indices.maxByOrNull {
                    classificationLogits[it]
                } ?: error("Classification output is empty")

                val predictedClassName = classificationLabels[predictedClassId]

                // Log the classification results.
                Log.d(
                    "DA3_CLASS",
                    "Classification logits: ${classificationLogits.contentToString()}"
                )

                Log.d(
                    "DA3_CLASS",
                    "Predicted class ID: $predictedClassId"
                )

                Log.d(
                    "DA3_CLASS",
                    "Predicted class name: $predictedClassName"
                )

                Log.d(
                    "DA3_SEG",
                    "Segmentation logits: ${segmentationLogits.size} values"
                )

                // Focal-length correction from the original DA3 Python pipeline
                val focal = (correctedFx + correctedFy) / 2f
                val depthScale = focal / 300f

                // Convert raw depth predictions using the DA3 scaling convention
                val metricDepthValues = FloatArray(depthValues.size) { i ->
                    depthValues[i] * depthScale
                }

                android.util.Log.d(
                    "DA3_METRIC",
                    "Corrected focal lengths: fx=$correctedFx, fy=$correctedFy"
                )

                android.util.Log.d(
                    "DA3_METRIC",
                    "Depth scaling factor: $depthScale"
                )

                android.util.Log.d(
                    "DA3_METRIC",
                    "Metric depth statistics: " +
                            "min=${metricDepthValues.minOrNull()}, " +
                            "max=${metricDepthValues.maxOrNull()}, " +
                            "mean=${metricDepthValues.average()}"
                )

                val depthBitmap = createDepthBitmap(
                    depthValues = metricDepthValues,
                    originalWidth = originalWidth,
                    originalHeight = originalHeight,
                    padTop = padTop,
                    padBottom = padBottom,
                    padLeft = padLeft,
                    padRight = padRight
                )

                // Prepare numerical metric depth at the original image resolution.
                val resizedMetricDepth = prepareMetricDepthMap(
                    metricDepthValues = metricDepthValues,
                    originalWidth = originalWidth,
                    originalHeight = originalHeight,
                    padTop = padTop,
                    padBottom = padBottom,
                    padLeft = padLeft,
                    padRight = padRight
                )

                // Convert segmentation logits to class IDs.
                val segmentationMask = prepareSegmentationMask(
                    segmentationLogits = segmentationLogits,
                    originalWidth = originalWidth,
                    originalHeight = originalHeight,
                    padTop = padTop,
                    padBottom = padBottom,
                    padLeft = padLeft,
                    padRight = padRight
                )

                // Encode the segmentation mask as an 8-bit grayscale PNG.
                val maskPngBytes = encodeSegmentationMaskPng(
                    segmentationMask = segmentationMask,
                    width = originalWidth,
                    height = originalHeight
                )

                Log.d(
                    "DA3_SEG",
                    "Segmentation PNG encoded successfully: ${maskPngBytes.size} bytes"
                )

                // Decode the PNG and verify that class IDs are preserved.
                val encodedMaskMat = MatOfByte(*maskPngBytes)

                try {
                    val decodedMaskMat = Imgcodecs.imdecode(
                        encodedMaskMat,
                        Imgcodecs.IMREAD_UNCHANGED
                    )

                    try {
                        check(!decodedMaskMat.empty()) {
                            "Failed to decode segmentation PNG"
                        }

                        check(decodedMaskMat.type() == CvType.CV_8UC1) {
                            "Segmentation PNG type mismatch: ${decodedMaskMat.type()}"
                        }

                        check(
                            decodedMaskMat.cols() == originalWidth &&
                                    decodedMaskMat.rows() == originalHeight
                        ) {
                            "Segmentation PNG dimensions mismatch"
                        }

                        val decodedMask = ByteArray(
                            originalWidth * originalHeight
                        )

                        decodedMaskMat.get(0, 0, decodedMask)

                        // PNG must preserve every class ID exactly.
                        check(decodedMask.contentEquals(segmentationMask)) {
                            "Segmentation PNG pixel values do not match original mask"
                        }

                        Log.d(
                            "DA3_SEG",
                            "Segmentation PNG verification successful: " +
                                    "type=CV_8UC1, " +
                                    "dimensions=${decodedMaskMat.cols()}x${decodedMaskMat.rows()}, " +
                                    "all class IDs preserved"
                        )

                    } finally {
                        decodedMaskMat.release()
                    }

                } finally {
                    encodedMaskMat.release()
                }

                Log.d(
                    "DA3_SEG",
                    "Segmentation mask generated: " +
                            "${originalWidth}x${originalHeight}, " +
                            "pixels=${segmentationMask.size}"
                )

                // Count pixels assigned to each segmentation class.
                val classCounts = IntArray(21)

                for (value in segmentationMask) {
                    val classId = value.toInt() and 0xFF

                    check(classId in 0..20) {
                        "Invalid segmentation class ID: $classId"
                    }

                    classCounts[classId]++
                }

                Log.d(
                    "DA3_SEG",
                    "Segmentation class distribution:"
                )

                for (classId in classCounts.indices) {
                    if (classCounts[classId] > 0) {
                        Log.d(
                            "DA3_SEG",
                            "Class $classId: ${classCounts[classId]} pixels"
                        )
                    }
                }

                // Log numerical depth information.
                Log.d(
                    "DA3_TIFF",
                    "Metric depth map: ${originalWidth}x${originalHeight}, " +
                            "pixels=${resizedMetricDepth.size}"
                )

                Log.d(
                    "DA3_TIFF",
                    "Resized metric depth: " +
                            "min=${resizedMetricDepth.minOrNull()}, " +
                            "max=${resizedMetricDepth.maxOrNull()}, " +
                            "mean=${resizedMetricDepth.average()}"
                )

                // Encode the numerical depth map as a Float32 TIFF.
                val tiffBytes = encodeMetricDepthTiff(
                    metricDepth = resizedMetricDepth,
                    width = originalWidth,
                    height = originalHeight
                )

                Log.d(
                    "DA3_TIFF",
                    "Float32 TIFF encoded successfully: ${tiffBytes.size} bytes"
                )

                // Decode the TIFF and verify its numerical values.
                val encodedMat = MatOfByte(*tiffBytes)

                try {
                    val decodedMat = Imgcodecs.imdecode(
                        encodedMat,
                        Imgcodecs.IMREAD_UNCHANGED
                    )

                    try {
                        check(!decodedMat.empty()) {
                            "Failed to decode the generated TIFF"
                        }

                        check(decodedMat.type() == CvType.CV_32FC1) {
                            "TIFF type mismatch: ${decodedMat.type()}"
                        }

                        check(
                            decodedMat.cols() == originalWidth &&
                                    decodedMat.rows() == originalHeight
                        ) {
                            "TIFF dimensions mismatch"
                        }

                        val decodedDepth = FloatArray(
                            originalWidth * originalHeight
                        )

                        decodedMat.get(0, 0, decodedDepth)

                        // Compare every pixel with the original numerical depth map.
                        var maxDifference = 0f

                        for (i in resizedMetricDepth.indices) {
                            val original = resizedMetricDepth[i]
                            val decoded = decodedDepth[i]

                            if (original.toRawBits() != decoded.toRawBits()) {
                                val difference = kotlin.math.abs(original - decoded)

                                if (difference > maxDifference) {
                                    maxDifference = difference
                                }
                            }
                        }

                        Log.d(
                            "DA3_TIFF",
                            "TIFF verification successful: " +
                                    "type=CV_32FC1, " +
                                    "dimensions=${decodedMat.cols()}x${decodedMat.rows()}, " +
                                    "maxDifference=$maxDifference"
                        )

                        val inputFileName = selectedImageFileName

                        if (inputFileName != null) {

                            // Save metric depth as Float32 TIFF.
                            saveMetricDepthTiff(
                                tiffBytes = tiffBytes,
                                inputFileName = inputFileName
                            )

                            // Save segmentation class IDs as 8-bit PNG.
                            saveSegmentationMaskPng(
                                pngBytes = maskPngBytes,
                                inputFileName = inputFileName
                            )

                            // Save the predicted classification name as TXT.
                            saveClassificationTxt(
                                predictedClassName = predictedClassName,
                                inputFileName = inputFileName
                            )

                            // Retrieve the parallel UFLD v2 lane detection result (computed concurrently with DA3)
                            val fullResLaneResult = try {
                                laneFuture.get()
                            } catch (e: Exception) {
                                Log.e("UFLDV2", "Failed to get parallel lane result", e)
                                Ufldv2Utils.LaneResult(ByteArray(originalWidth * originalHeight), ArrayList(), ArrayList(), 0f, 0f)
                            }

                            // Stop full-res lane mask at distance ~15m using resizedMetricDepth
                            val fullResMaskBytes = fullResLaneResult.maskBytes
                            for (i in fullResMaskBytes.indices) {
                                val maskVal = fullResMaskBytes[i].toInt() and 0xFF
                                val depth = if (i < resizedMetricDepth.size) resizedMetricDepth[i] else 0f
                                if (maskVal > 0 && (depth <= 0f || depth > 15.0f)) {
                                    fullResMaskBytes[i] = 0
                                }
                            }

                            val detectionPreview = selectedBitmap!!.copy(Bitmap.Config.ARGB_8888, true)
                            drawDetectedLaneLines(Canvas(detectionPreview), fullResLaneResult)
                            runOnUiThread {
                                imagePreview.setImageBitmap(detectionPreview)
                                imagePreview.contentDescription = "Detected lane lines: red left, cyan right"
                                textImageInfo.text = "Detected lanes: red left (${fullResLaneResult.detectedLeftLane.size} points), " +
                                    "cyan right (${fullResLaneResult.detectedRightLane.size} points). Dots show detections."
                            }

                            if (outputFolderUri != null) {
                                saveBikeLaneOverlayPng(
                                    originalBitmap = selectedBitmap!!,
                                    laneResult = fullResLaneResult,
                                    inputFileName = inputFileName,
                                    detectionOnly = true
                                )
                                saveBikeLaneOverlayPng(
                                    originalBitmap = selectedBitmap!!,
                                    laneResult = fullResLaneResult,
                                    inputFileName = inputFileName
                                )
                            }

                            // Generate 504x504 ROI mask directly from fullResLaneResult lanes (avoids re-running ONNX model)
                            val roiMaskBytes = try {
                                Ufldv2Utils.generateRoiMaskFromLanes(
                                    selectedBitmap!!,
                                    fullResLaneResult.leftLane,
                                    fullResLaneResult.rightLane,
                                    504,
                                    504
                                )
                            } catch (e: Exception) {
                                Log.e("UFLDV2", "Failed to generate bike lane ROI mask", e)
                                ByteArray(504 * 504) { 255.toByte() }
                            }

                            // Stop 504x504 ROI mask at distance ~10m using metricDepthValues
                            for (i in roiMaskBytes.indices) {
                                val maskVal = roiMaskBytes[i].toInt() and 0xFF
                                val depth = metricDepthValues[i]
                                if (maskVal > 0 && (depth <= 0f || depth > 10.0f)) {
                                    roiMaskBytes[i] = 0
                                }
                            }

                            val maskedDepthValues = FloatArray(metricDepthValues.size) { i ->
                                val maskVal = roiMaskBytes[i].toInt() and 0xFF
                                if (maskVal > 0) metricDepthValues[i] else 0f
                            }

                            val modelIntrinsics = CameraIntrinsics(correctedFx, correctedFy, correctedCx, correctedCy)
                            val planeResult = RansacUtils.estimateGroundPlane(
                                metricDepth = maskedDepthValues,
                                width = 504,
                                height = 504,
                                intrinsics = modelIntrinsics
                            )

                            Log.d("DA3_RANSAC", "Plane equation: ${planeResult.a}x + ${planeResult.b}y + ${planeResult.c}z + ${planeResult.d} = 0")
                            Log.d("DA3_RANSAC", "Inlier ratio: ${planeResult.inlierRatio}")
                            Log.d("DA3_RANSAC", "Tilt: ${planeResult.tiltDeg}°, Roll: ${planeResult.rollDeg}°, Pitch: ${planeResult.pitchDeg}°")

                            // Object Detection Pipeline inside Bike Lane ROI (Ground Plane Removed)
                            val detectedObjects = ObjectDetector.detectObjects(
                                metricDepth = metricDepthValues,
                                width = 504,
                                height = 504,
                                intrinsics = modelIntrinsics,
                                plane = planeResult,
                                roiMaskBytes = roiMaskBytes
                            )

                            Log.d("DA3_OBJECTS", "Detected ${detectedObjects.size} objects in bike lane ROI")
                            for ((idx, obj) in detectedObjects.withIndex()) {
                                Log.d("DA3_OBJECTS", "Object $idx: distance=${obj.distanceMeters}m, height=${obj.heightMeters}m, pixels=${obj.pixelCount}")
                            }

                            val objectsOverlayBitmap = ObjectDetector.drawObjectsOverlay(
                                selectedBitmap!!,
                                detectedObjects
                            )

                            runOnUiThread {
                                imagePreview.setImageBitmap(objectsOverlayBitmap)
                                imagePreview.contentDescription = "Detected objects in bike lane ROI with distance and height"
                                textImageInfo.text = "Detected ${detectedObjects.size} objects in bike lane. Red boxes show objects with distance & height."
                            }

                            if (outputFolderUri != null) {
                                try {
                                    val baseName = inputFileName.substringBeforeLast('.', inputFileName)
                                    val objImageName = "${baseName}_objects_overlay.png"
                                    val folder = DocumentFile.fromTreeUri(this, outputFolderUri!!)
                                    if (folder != null) {
                                        folder.findFile(objImageName)?.delete()
                                        val outputFile = folder.createFile("image/png", objImageName)
                                        if (outputFile != null) {
                                            contentResolver.openOutputStream(outputFile.uri)?.use { output ->
                                                objectsOverlayBitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
                                            }
                                            Log.d("DA3_OBJECTS", "Saved objects overlay PNG successfully")
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e("DA3_OBJECTS", "Failed to save objects overlay PNG", e)
                                }
                            }

                            if (outputFolderUri != null) {
                                runOnUiThread {
                                    androidx.appcompat.app.AlertDialog.Builder(this)
                                        .setTitle("Save Point Cloud (.ply)")
                                        .setMessage("RANSAC ground plane fitted successfully (Inlier ratio: ${(planeResult.inlierRatio * 100).toInt()}%).\nWould you like to save the 3D point cloud (.ply) to the selected output folder?")
                                        .setPositiveButton("Save PLY") { _, _ ->
                                            Thread {
                                                try {
                                                    val plyBytes = RansacUtils.exportPly(
                                                        metricDepth = maskedDepthValues,
                                                        width = 504,
                                                        height = 504,
                                                        intrinsics = modelIntrinsics,
                                                        plane = planeResult
                                                    )
                                                    val baseName = inputFileName.substringBeforeLast('.', inputFileName)
                                                    val plyName = "${baseName}_plane.ply"
                                                    val folder = DocumentFile.fromTreeUri(this, outputFolderUri!!)
                                                    if (folder != null) {
                                                        folder.findFile(plyName)?.delete()
                                                        val outputFile = folder.createFile("application/octet-stream", plyName)
                                                        if (outputFile != null) {
                                                            contentResolver.openOutputStream(outputFile.uri)?.use { output ->
                                                                output.write(plyBytes)
                                                            }
                                                            runOnUiThread {
                                                                Toast.makeText(this, "Saved $plyName successfully", Toast.LENGTH_SHORT).show()
                                                            }
                                                        }
                                                    }
                                                } catch (e: Exception) {
                                                    runOnUiThread {
                                                        Toast.makeText(this, "Failed to save PLY: ${e.message}", Toast.LENGTH_LONG).show()
                                                    }
                                                }
                                            }.start()
                                        }
                                        .setNegativeButton("No thanks", null)
                                        .show()
                                }
                            }

                        } else {

                            Log.e(
                                "DA3_CLASS",
                                "Cannot save outputs: original image filename is missing"
                            )
                        }

                    } finally {
                        decodedMat.release()
                    }

                } finally {
                    encodedMat.release()
                }

                val elapsedSeconds =
                    (System.currentTimeMillis() - startTime) / 1000.0

                android.util.Log.d(
                    "DA3_INFERENCE",
                    "Inference completed in $elapsedSeconds seconds"
                )

                android.util.Log.d(
                    "DA3_INFERENCE",
                    "Number of outputs: ${outputs.size}"
                )

                outputs.forEachIndexed { index, output ->

                    val tensor = output.toTensor()
                    val values = tensor.dataAsFloatArray

                    android.util.Log.d(
                        "DA3_INFERENCE",
                        "Output $index shape: ${tensor.shape().contentToString()}"
                    )

                    android.util.Log.d(
                        "DA3_INFERENCE",
                        "Output $index elements: ${tensor.numel()}"
                    )

                    var minValue = Float.POSITIVE_INFINITY
                    var maxValue = Float.NEGATIVE_INFINITY
                    var sum = 0.0
                    var finiteCount = 0

                    for (value in values) {
                        if (value.isFinite()) {
                            if (value < minValue) {
                                minValue = value
                            }

                            if (value > maxValue) {
                                maxValue = value
                            }

                            sum += value.toDouble()
                            finiteCount++
                        }
                    }

                    val nonFiniteCount = values.size - finiteCount

                    val minimum: Float? = if (finiteCount > 0) minValue else null
                    val maximum: Float? = if (finiteCount > 0) maxValue else null
                    val mean: Double = if (finiteCount > 0) {
                        sum / finiteCount
                    } else {
                        Double.NaN
                    }

                    android.util.Log.d(
                        "DA3_INFERENCE",
                        "Output $index statistics: min=$minimum, max=$maximum, mean=$mean, nonFinite=$nonFiniteCount"
                    )

                    if (index == 2) {

                        android.util.Log.d(
                            "DA3_INFERENCE",
                            "Classification logits: ${values.contentToString()}"
                        )

                        val predictedClass = values.indices.maxByOrNull { values[it] }

                        android.util.Log.d(
                            "DA3_INFERENCE",
                            "Predicted classification index: $predictedClass"
                        )
                    }
                }

                runOnUiThread {

                    depthPreview.setImageBitmap(depthBitmap)

                    depthPreview.visibility = View.VISIBLE
                    textDepthTitle.visibility = View.VISIBLE

                    Toast.makeText(
                        this,
                        "DA3 inference completed successfully!",
                        Toast.LENGTH_LONG
                    ).show()
                }

            } catch (e: Exception) {

                android.util.Log.e(
                    "DA3_INFERENCE",
                    "Inference failed",
                    e
                )

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "DA3 inference failed: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }.start()
    }

    private fun tensorSha256(tensor: FloatArray): String {

        val buffer = ByteBuffer.allocate(tensor.size * 4)
            .order(ByteOrder.LITTLE_ENDIAN)

        for (value in tensor) {
            buffer.putFloat(value)
        }

        val digest = MessageDigest.getInstance("SHA-256")
            .digest(buffer.array())

        return digest.joinToString("") {
            "%02x".format(it.toInt() and 0xFF)
        }
    }

    private val pickImageLauncher =
        registerForActivityResult(
            ActivityResultContracts.OpenDocument()
        ) { uri: Uri? ->

            if (uri != null) {

                selectedImageFileName = getFileNameFromUri(uri)

                Log.d(
                    "DA3_TIFF",
                    "Selected image filename: $selectedImageFileName"
                )

                loadImage(uri)
            }
        }

    private val cameraLauncher =
        registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->

            if (result.resultCode == RESULT_OK) {

                capturedCameraIntrinsics = null

                val photoPath =
                    result.data?.getStringExtra("captured_photo_path")

                val capturedCameraId =
                    result.data?.getStringExtra("captured_camera_id")

                val jpegWidthForCalibration: Int
                val jpegHeightForCalibration: Int
                val jpegOrientationForCalibration: Int

                if (photoPath != null && File(photoPath).exists()) {

                    val options = BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }

                    BitmapFactory.decodeFile(photoPath, options)

                    val exif = ExifInterface(photoPath)

                    jpegWidthForCalibration = options.outWidth
                    jpegHeightForCalibration = options.outHeight

                    jpegOrientationForCalibration = exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_UNDEFINED
                    )

                } else {
                    jpegWidthForCalibration = -1
                    jpegHeightForCalibration = -1
                    jpegOrientationForCalibration = ExifInterface.ORIENTATION_UNDEFINED
                }

                Log.d(
                    "DA3_CAMERA_METADATA",
                    "Received camera ID from CameraActivity: $capturedCameraId"
                )

                val capturedSensorTimestamp =
                    result.data?.getLongExtra("captured_sensor_timestamp", -1L) ?: -1L

                Log.d(
                    "DA3_CAMERA_METADATA",
                    "Received sensor timestamp: $capturedSensorTimestamp"
                )

                val cameraData = result.data

                val cropLeft = cameraData?.getIntExtra("captured_crop_left", -1) ?: -1
                val cropTop = cameraData?.getIntExtra("captured_crop_top", -1) ?: -1
                val cropRight = cameraData?.getIntExtra("captured_crop_right", -1) ?: -1
                val cropBottom = cameraData?.getIntExtra("captured_crop_bottom", -1) ?: -1

                Log.d(
                    "DA3_CAMERA_METADATA",
                    "Received crop region: " +
                            "left=$cropLeft, top=$cropTop, " +
                            "right=$cropRight, bottom=$cropBottom"
                )

                val distortionMode = cameraData?.getIntExtra(
                    "captured_distortion_mode",
                    -1
                ) ?: -1

                val zoomRatio = cameraData?.getFloatExtra(
                    "captured_zoom_ratio",
                    Float.NaN
                ) ?: Float.NaN

                Log.d(
                    "DA3_CAMERA_METADATA",
                    "Received distortion correction mode: $distortionMode"
                )

                Log.d(
                    "DA3_CAMERA_METADATA",
                    "Received zoom ratio: $zoomRatio"
                )

                if (capturedCameraId != null) {

                    try {
                        val cameraManager = getSystemService(
                            CAMERA_SERVICE
                        ) as CameraManager

                        val characteristics = cameraManager.getCameraCharacteristics(
                            capturedCameraId
                        )

                        val distortion = characteristics.get(
                            CameraCharacteristics.LENS_DISTORTION
                        )

                        val availableDistortionModes = characteristics.get(
                            CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES
                        )

                        Log.d(
                            "DA3_DISTORTION",
                            "Lens distortion coefficients: ${distortion?.contentToString()}"
                        )

                        Log.d(
                            "DA3_DISTORTION",
                            "Available distortion correction modes: " +
                                    "${availableDistortionModes?.contentToString()}"
                        )

                        Log.d(
                            "DA3_DISTORTION",
                            "Captured distortion correction mode: $distortionMode"
                        )

                        if (android.os.Build.VERSION.SDK_INT >= 28) {

                            val physicalIds = characteristics.physicalCameraIds

                            for (physicalId in physicalIds) {
                                try {
                                    val physicalCharacteristics =
                                        cameraManager.getCameraCharacteristics(physicalId)

                                    val calibration = physicalCharacteristics.get(
                                        CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
                                    )

                                    val activeArray = physicalCharacteristics.get(
                                        CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
                                    )

                                    Log.d(
                                        "DA3_PHYSICAL_CALIBRATION",
                                        "Physical camera $physicalId: " +
                                                "intrinsics=${calibration?.contentToString()}, " +
                                                "activeArray=$activeArray"
                                    )

                                } catch (e: Exception) {
                                    Log.d(
                                        "DA3_PHYSICAL_CALIBRATION",
                                        "Cannot inspect physical camera $physicalId: ${e.message}"
                                    )
                                }
                            }
                        }

                        val calibration = characteristics.get(
                            CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
                        )

                        val activeArray = characteristics.get(
                            CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
                        )

                        val preCorrectionArray = characteristics.get(
                            CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE
                        )

                        Log.d(
                            "DA3_CAMERA_METADATA",
                            "Sensor active array: $activeArray"
                        )

                        Log.d(
                            "DA3_CAMERA_METADATA",
                            "Sensor pre-correction active array: $preCorrectionArray"
                        )

                        val capabilities = characteristics.get(
                            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES
                        )

                        val hasMotionTracking = capabilities?.contains(
                            CameraCharacteristics
                                .REQUEST_AVAILABLE_CAPABILITIES_MOTION_TRACKING
                        ) == true

                        Log.d(
                            "DA3_CAMERA_METADATA",
                            "Motion tracking capability: $hasMotionTracking"
                        )

                        val isLogicalMultiCamera = capabilities?.contains(
                            CameraCharacteristics
                                .REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA
                        ) == true

                        Log.d(
                            "DA3_CAMERA_METADATA",
                            "Is logical multi-camera: $isLogicalMultiCamera"
                        )

                        if (android.os.Build.VERSION.SDK_INT >= 28) {

                            val physicalCameraIds = characteristics.physicalCameraIds

                            Log.d(
                                "DA3_CAMERA_METADATA",
                                "Physical camera IDs: $physicalCameraIds"
                            )
                        }

                        val sensorOrientation = characteristics.get(
                            CameraCharacteristics.SENSOR_ORIENTATION
                        )

                        Log.d(
                            "DA3_CAMERA_METADATA",
                            "Camera sensor orientation: $sensorOrientation degrees"
                        )

                        if (calibration != null && calibration.size >= 5) {

                            Log.d(
                                "DA3_CAMERA_METADATA",
                                "Camera intrinsic calibration: " +
                                        "fx=${calibration[0]}, " +
                                        "fy=${calibration[1]}, " +
                                        "cx=${calibration[2]}, " +
                                        "cy=${calibration[3]}, " +
                                        "skew=${calibration[4]}"
                            )

                            // Intrinsics reported for the original, unrotated image
                            val originalFx = calibration[0]
                            val originalFy = calibration[1]
                            val originalCx = calibration[2]
                            val originalCy = calibration[3]

                            if (
                                jpegWidthForCalibration > 0 &&
                                jpegHeightForCalibration > 0 &&
                                jpegOrientationForCalibration == ExifInterface.ORIENTATION_ROTATE_90
                            ) {

                                val rotatedFx = originalFy
                                val rotatedFy = originalFx
                                val rotatedCx = (jpegHeightForCalibration - 1) - originalCy
                                val rotatedCy = originalCx

                                capturedCameraIntrinsics = CameraIntrinsics(
                                    fx = rotatedFx,
                                    fy = rotatedFy,
                                    cx = rotatedCx,
                                    cy = rotatedCy
                                )

                                findViewById<EditText>(R.id.inputFx).setText(rotatedFx.toString())
                                findViewById<EditText>(R.id.inputFy).setText(rotatedFy.toString())
                                findViewById<EditText>(R.id.inputCx).setText(rotatedCx.toString())
                                findViewById<EditText>(R.id.inputCy).setText(rotatedCy.toString())

                                Log.d(
                                    "DA3_AUTO_INTRINSICS",
                                    "Stored automatic camera intrinsics: " +
                                            "fx=$rotatedFx, fy=$rotatedFy, " +
                                            "cx=$rotatedCx, cy=$rotatedCy"
                                )

                                Log.d(
                                    "DA3_ROTATED_INTRINSICS",
                                    "Rotated intrinsics: " +
                                            "fx=$rotatedFx, " +
                                            "fy=$rotatedFy, " +
                                            "cx=$rotatedCx, " +
                                            "cy=$rotatedCy"
                                )

                            } else {

                                capturedCameraIntrinsics = CameraIntrinsics(
                                    fx = originalFx,
                                    fy = originalFy,
                                    cx = originalCx,
                                    cy = originalCy
                                )

                                findViewById<EditText>(R.id.inputFx).setText(originalFx.toString())
                                findViewById<EditText>(R.id.inputFy).setText(originalFy.toString())
                                findViewById<EditText>(R.id.inputCx).setText(originalCx.toString())
                                findViewById<EditText>(R.id.inputCy).setText(originalCy.toString())

                                Log.d(
                                    "DA3_AUTO_INTRINSICS",
                                    "Stored automatic camera intrinsics (unrotated): " +
                                            "fx=$originalFx, fy=$originalFy, " +
                                            "cx=$originalCx, cy=$originalCy"
                                )
                            }

                        } else {

                            Log.w(
                                "DA3_CAMERA_METADATA",
                                "Intrinsic calibration unavailable for camera ID $capturedCameraId"
                            )
                        }

                    } catch (e: Exception) {

                        Log.e(
                            "DA3_CAMERA_METADATA",
                            "Failed to retrieve camera calibration",
                            e
                        )
                    }
                }

                if (photoPath != null) {

                    val jpegOptions = BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }

                    BitmapFactory.decodeFile(photoPath, jpegOptions)

                    val jpegExif = ExifInterface(photoPath)

                    val jpegOrientation = jpegExif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_UNDEFINED
                    )

                    val originalJpegWidth = jpegOptions.outWidth
                    val originalJpegHeight = jpegOptions.outHeight

                    if (originalJpegWidth > 0 && originalJpegHeight > 0) {
                        Log.d(
                            "DA3_JPEG_CALIBRATION",
                            "Valid JPEG geometry available for camera calibration"
                        )
                    } else {
                        Log.e(
                            "DA3_JPEG_CALIBRATION",
                            "Invalid JPEG dimensions: " +
                                    "${originalJpegWidth}x${originalJpegHeight}"
                        )
                    }

                    Log.d(
                        "DA3_JPEG_CALIBRATION",
                        "Original JPEG geometry: " +
                                "width=$originalJpegWidth, " +
                                "height=$originalJpegHeight, " +
                                "orientation=$jpegOrientation"
                    )

                    Log.d(
                        "DA3_IMAGE_GEOMETRY",
                        "Stored JPEG dimensions: " +
                                "${jpegOptions.outWidth} x ${jpegOptions.outHeight}"
                    )

                    Log.d(
                        "DA3_IMAGE_GEOMETRY",
                        "Stored JPEG EXIF orientation: $jpegOrientation"
                    )

                    Log.d(
                        "DA3_CAMERA",
                        "MainActivity received photograph: $photoPath"
                    )

                    val photoFile = File(photoPath)

                    if (photoFile.exists()) {

                        val capturedBitmap = loadCorrectlyOrientedBitmap(photoPath)

                        if (capturedBitmap != null) {

                            // Store the correctly oriented image for DA3 inference.
                            selectedBitmap = capturedBitmap

                            Log.d(
                                "DA3_IMAGE_GEOMETRY",
                                "Corrected bitmap dimensions: " +
                                        "${capturedBitmap.width} x ${capturedBitmap.height}"
                            )

                            // Preserve the original camera filename for output files.
                            selectedImageFileName = photoFile.name

                            // Display the same bitmap that will be used for inference.
                            imagePreview.setImageBitmap(capturedBitmap)

                            // Clear the previous depth prediction.
                            depthPreview.setImageDrawable(null)
                            depthPreview.visibility = View.GONE
                            textDepthTitle.visibility = View.GONE

                            // Update the image information.
                            textImageInfo.text =
                                "Image size: ${capturedBitmap.width} × ${capturedBitmap.height}"

                            Log.d(
                                "DA3_CAMERA",
                                "Captured photograph ready for inference: " +
                                        "${capturedBitmap.width}x${capturedBitmap.height}"
                            )

                            Log.d(
                                "DA3_CAMERA",
                                "Selected camera filename: $selectedImageFileName"
                            )
                        } else {
                            Log.e(
                                "DA3_CAMERA",
                                "Failed to decode captured photograph"
                            )
                        }

                    } else {
                        Log.e(
                            "DA3_CAMERA",
                            "Captured photograph file does not exist"
                        )
                    }
                } else {
                    Log.e(
                        "DA3_CAMERA",
                        "Camera result does not contain a photograph path"
                    )
                }
            }
        }

    private val outputFolderPicker =
        registerForActivityResult(
            ActivityResultContracts.OpenDocumentTree()
        ) { uri ->

            if (uri != null) {

                outputFolderUri = uri

                // Keep access to the selected folder after app restarts
                contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or
                            Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )

                // Save the folder URI in app preferences
                getSharedPreferences("DA3CameraPrefs", MODE_PRIVATE)
                    .edit()
                    .putString("outputFolderUri", uri.toString())
                    .apply()

                textOutputFolder.text = "Output folder selected"

                android.util.Log.d(
                    "DA3_STORAGE",
                    "Selected output folder: $uri"
                )

                Toast.makeText(
                    this,
                    "Output folder selected successfully",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

    private val pickIntrinsicsFileLauncher =
        registerForActivityResult(
            ActivityResultContracts.OpenDocument()
        ) { uri: Uri? ->
            if (uri != null) {
                try {
                    val content = contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                    if (content != null) {
                        val regex = Regex("[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?")
                        val matches = regex.findAll(content).map { it.value }.toList()
                        if (matches.size >= 9) {
                            val fx = matches[0].toFloatOrNull()
                            val cx = matches[2].toFloatOrNull()
                            val fy = matches[4].toFloatOrNull()
                            val cy = matches[5].toFloatOrNull()

                            if (fx != null && fy != null && cx != null && cy != null) {
                                findViewById<EditText>(R.id.inputFx).setText(fx.toString())
                                findViewById<EditText>(R.id.inputFy).setText(fy.toString())
                                findViewById<EditText>(R.id.inputCx).setText(cx.toString())
                                findViewById<EditText>(R.id.inputCy).setText(cy.toString())

                                Toast.makeText(this, "Intrinsics imported successfully!", Toast.LENGTH_SHORT).show()
                            } else {
                                Toast.makeText(this, "Failed to parse intrinsic values", Toast.LENGTH_LONG).show()
                            }
                        } else if (matches.size >= 4) {
                            val fx = matches[0].toFloatOrNull()
                            val fy = matches[1].toFloatOrNull()
                            val cx = matches[2].toFloatOrNull()
                            val cy = matches[3].toFloatOrNull()

                            if (fx != null && fy != null && cx != null && cy != null) {
                                findViewById<EditText>(R.id.inputFx).setText(fx.toString())
                                findViewById<EditText>(R.id.inputFy).setText(fy.toString())
                                findViewById<EditText>(R.id.inputCx).setText(cx.toString())
                                findViewById<EditText>(R.id.inputCy).setText(cy.toString())

                                Toast.makeText(this, "Intrinsics imported successfully!", Toast.LENGTH_SHORT).show()
                            } else {
                                Toast.makeText(this, "Failed to parse intrinsic values", Toast.LENGTH_LONG).show()
                            }
                        } else {
                            Toast.makeText(this, "Expected at least 4 or 9 numeric values, found ${matches.size}", Toast.LENGTH_LONG).show()
                        }
                    }
                } catch (e: Exception) {
                    Toast.makeText(this, "Error reading file: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

    private fun encodeSegmentationMaskPng(
        segmentationMask: ByteArray,
        width: Int,
        height: Int
    ): ByteArray {

        require(width > 0 && height > 0) {
            "Invalid segmentation dimensions"
        }

        require(segmentationMask.size == width * height) {
            "Segmentation mask size does not match image dimensions"
        }

        // Single-channel, 8-bit unsigned image.
        // Pixel values represent class IDs from 0 to 20.
        val maskMat = Mat(height, width, CvType.CV_8UC1)
        val encoded = MatOfByte()

        try {
            maskMat.put(0, 0, segmentationMask)

            val success = Imgcodecs.imencode(
                ".png",
                maskMat,
                encoded
            )

            check(success) {
                "OpenCV failed to encode segmentation PNG"
            }

            return encoded.toArray()

        } finally {
            maskMat.release()
            encoded.release()
        }
    }

    private fun encodeMetricDepthTiff(
        metricDepth: FloatArray,
        width: Int,
        height: Int
    ): ByteArray {

        require(metricDepth.size == width * height) {
            "Depth array size does not match image dimensions"
        }

        // Single-channel, 32-bit floating-point image.
        // Each pixel contains the predicted depth in meters.
        val depthMat = Mat(height, width, CvType.CV_32FC1)
        val encoded = MatOfByte()

        try {
            depthMat.put(0, 0, metricDepth)

            val success = Imgcodecs.imencode(
                ".tiff",
                depthMat,
                encoded
            )

            check(success) {
                "OpenCV failed to encode Float32 TIFF"
            }

            return encoded.toArray()

        } finally {
            depthMat.release()
            encoded.release()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            val modelName = "da3_baseline72_fixed504.pte"

            val modelSize = assets.openFd(modelName).use {
                it.length
            }

            android.util.Log.d(
                "DA3_MODEL",
                "Model found in assets: $modelSize bytes"
            )

        } catch (e: Exception) {

            android.util.Log.e(
                "DA3_MODEL",
                "Could not access model asset",
                e
            )
        }

        Thread {
            try {
                android.util.Log.d(
                    "DA3_MODEL",
                    "Starting model copy..."
                )

                val startTime = System.currentTimeMillis()

                val modelFile = DA3ModelLoader.prepareModelFile(
                    applicationContext
                )

                Ufldv2Utils.initialize(applicationContext)

                val elapsedSeconds =
                    (System.currentTimeMillis() - startTime) / 1000.0

                android.util.Log.d(
                    "DA3_MODEL",
                    "Model copy completed successfully!"
                )

                android.util.Log.d(
                    "DA3_MODEL",
                    "Model path: ${modelFile.absolutePath}"
                )

                android.util.Log.d(
                    "DA3_MODEL",
                    "Model size: ${modelFile.length()} bytes"
                )

                android.util.Log.d(
                    "DA3_MODEL",
                    "Copy time: $elapsedSeconds seconds"
                )

                android.util.Log.d(
                    "DA3_MODEL",
                    "Attempting to load model with ExecuTorch..."
                )

                val startLoadTime = System.currentTimeMillis()

                val loadedModule = DA3ModelLoader.loadModel(modelFile)

                val loadSeconds =
                    (System.currentTimeMillis() - startLoadTime) / 1000.0

                da3Module = loadedModule

                android.util.Log.d(
                    "DA3_MODEL",
                    "ExecuTorch Module.load() completed successfully"
                )

                android.util.Log.d(
                    "DA3_MODEL",
                    "Loading time: $loadSeconds seconds"
                )

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "ExecuTorch model loaded successfully!",
                        Toast.LENGTH_LONG
                    ).show()
                }

            } catch (e: Exception) {

                android.util.Log.e(
                    "DA3_MODEL",
                    "Model preparation failed",
                    e
                )

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Model preparation failed: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }.start()

        val openCvLoaded = org.opencv.android.OpenCVLoader.initLocal()

        android.util.Log.d(
            "DA3_DEBUG",
            "OpenCV initialized: $openCvLoaded"
        )

        if (openCvLoaded) {

            android.util.Log.d(
                "DA3_DEBUG",
                "Before Mat creation"
            )

            val testMat = org.opencv.core.Mat(
                10,
                10,
                org.opencv.core.CvType.CV_8UC3
            )

            android.util.Log.d(
                "DA3_DEBUG",
                "Mat created: ${testMat.rows()} x ${testMat.cols()}"
            )

            testMat.release()

            android.util.Log.d(
                "DA3_DEBUG",
                "Mat released successfully"
            )
        }

        imagePreview = findViewById(R.id.imagePreview)
        textImageInfo = findViewById(R.id.textImageInfo)

        depthPreview = findViewById(R.id.depthPreview)
        textDepthTitle = findViewById(R.id.textDepthTitle)
        textOutputFolder = findViewById(R.id.textOutputFolder)

        val savedFolderUri =
            getSharedPreferences("DA3CameraPrefs", MODE_PRIVATE)
                .getString("outputFolderUri", null)

        if (savedFolderUri != null) {

            outputFolderUri = Uri.parse(savedFolderUri)

            textOutputFolder.text = "Output folder selected"
        }

        findViewById<Button>(R.id.btnSelectOutputFolder)
            .setOnClickListener {

                outputFolderPicker.launch(null)
            }

        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImageLauncher.launch((arrayOf("image/*")))
        }

        findViewById<Button>(R.id.btnImportIntrinsics).setOnClickListener {
            pickIntrinsicsFileLauncher.launch(arrayOf("text/plain", "*/*"))
        }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {

            if (
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                Log.d("DA3_CAMERA", "Opening CameraActivity")

                cameraLauncher.launch(
                    Intent(this, CameraActivity::class.java)
                )

            } else {
                Log.d("DA3_CAMERA", "Camera permission not granted")

                requestCameraPermissionLauncher.launch(
                    Manifest.permission.CAMERA
                )
            }
        }


        findViewById<Button>(R.id.btnProcess).setOnClickListener {

            val fx = findViewById<EditText>(R.id.inputFx)
                .text.toString().toFloatOrNull()

            val fy = findViewById<EditText>(R.id.inputFy)
                .text.toString().toFloatOrNull()

            val cx = findViewById<EditText>(R.id.inputCx)
                .text.toString().toFloatOrNull()

            val cy = findViewById<EditText>(R.id.inputCy)
                .text.toString().toFloatOrNull()

            if (
                capturedCameraIntrinsics == null &&
                (
                        fx == null || fy == null || cx == null || cy == null ||
                                !fx.isFinite() || !fy.isFinite() ||
                                !cx.isFinite() || !cy.isFinite() ||
                                fx <= 0f || fy <= 0f
                        )
            ) {
                Toast.makeText(
                    this,
                    "Please enter valid camera intrinsics",
                    Toast.LENGTH_LONG
                ).show()
            }else {

                val bitmap = selectedBitmap

                if (bitmap == null) {
                    Toast.makeText(
                        this,
                        "Please select an image first",
                        Toast.LENGTH_LONG
                    ).show()

                } else {

                    try {

                        val intrinsics = capturedCameraIntrinsics ?: CameraIntrinsics(
                            fx = requireNotNull(fx),
                            fy = requireNotNull(fy),
                            cx = requireNotNull(cx),
                            cy = requireNotNull(cy)
                        )

                        Log.d(
                            "DA3_INTRINSICS_SOURCE",
                            if (capturedCameraIntrinsics != null) {
                                "Using automatic camera intrinsics: " +
                                        "fx=${intrinsics.fx}, fy=${intrinsics.fy}, " +
                                        "cx=${intrinsics.cx}, cy=${intrinsics.cy}"
                            } else {
                                "Using manually entered intrinsics: " +
                                        "fx=${intrinsics.fx}, fy=${intrinsics.fy}, " +
                                        "cx=${intrinsics.cx}, cy=${intrinsics.cy}"
                            }
                        )

                        android.util.Log.d(
                            "DA3_DEBUG",
                            "Starting DA3 preprocessing"
                        )

                        val result = DA3Preprocessor.preprocess(
                            bitmap,
                            intrinsics
                        )

                        val fingerprint = tensorSha256(result.inputTensor)

                        android.util.Log.d(
                            "DA3_HASH",
                            "Android tensor SHA-256: $fingerprint"
                        )

                        val tensor = result.inputTensor

                        val tensorMin = tensor.minOrNull()
                        val tensorMax = tensor.maxOrNull()
                        val tensorMean = tensor.average()

                        val nonZeroCount = tensor.count { it != 0.0f }

                        android.util.Log.d(
                            "DA3_TENSOR",
                            """
                            Tensor validation:
                            Shape: [1, 3, 504, 504]
                            Elements: ${tensor.size}
                            Minimum: $tensorMin
                            Maximum: $tensorMax
                            Mean: $tensorMean
                            Non-zero elements: $nonZeroCount
                            """.trimIndent()
                        )

                        android.util.Log.d(
                            "DA3_DEBUG",
                            "DA3 preprocessing completed successfully"
                        )

                        val k = result.correctedIntrinsics

                        val message = """
                            Preprocessing successful!
                        
                            Tensor: [1, 3, 504, 504]
                        
                            Original image:
                            ${result.originalWidth} × ${result.originalHeight}
                        
                            Processed image:
                            ${result.processedWidth} × ${result.processedHeight}
                        
                            Padding:
                            Left: ${result.padLeft}
                            Right: ${result.padRight}
                            Top: ${result.padTop}
                            Bottom: ${result.padBottom}
                        
                            Corrected intrinsics:
                            fx = ${k.fx}
                            fy = ${k.fy}
                            cx = ${k.cx}
                            cy = ${k.cy}
                        """.trimIndent()

                        android.util.Log.d(
                            "DA3_DEBUG",
                            message
                        )

                        android.app.AlertDialog.Builder(this)
                            .setTitle("DA3 Preprocessing")
                            .setMessage(message)
                            .setPositiveButton("OK", null)
                            .show()

                        runDA3Inference(
                            input = result.inputTensor,
                            originalWidth = result.originalWidth,
                            originalHeight = result.originalHeight,
                            padTop = result.padTop,
                            padBottom = result.padBottom,
                            padLeft = result.padLeft,
                            padRight = result.padRight,
                            correctedFx = result.correctedIntrinsics.fx,
                            correctedFy = result.correctedIntrinsics.fy,
                            correctedCx = result.correctedIntrinsics.cx,
                            correctedCy = result.correctedIntrinsics.cy
                        )

                    } catch (e: Exception) {

                        Toast.makeText(
                            this,
                            "Preprocessing error: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()

                        android.util.Log.e(
                            "DA3_PREPROCESS",
                            "Preprocessing failed",
                            e
                        )
                    }
                }
            }
        }

    }

    private fun loadImage(uri: Uri) {
        try {
            val bitmap: Bitmap =
                if (android.os.Build.VERSION.SDK_INT >= 28) {
                    val source = ImageDecoder.createSource(
                        contentResolver,
                        uri
                    )
                    ImageDecoder.decodeBitmap(source) { decoder, _, _ ->

                        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE

                    }
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(
                        contentResolver,
                        uri
                    )
                }

            capturedCameraIntrinsics = null
            selectedBitmap = bitmap
            imagePreview.setImageBitmap(bitmap)

            depthPreview.setImageDrawable(null)
            depthPreview.visibility = View.GONE
            textDepthTitle.visibility = View.GONE

            textImageInfo.text =
                "Image size: ${bitmap.width} × ${bitmap.height}"

        } catch (e: Exception) {
            Toast.makeText(
                this,
                "Error loading image: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }
}
