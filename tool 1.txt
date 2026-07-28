<div align="center">

# 🖼️ Digital Image Processing & Computer Vision
## Assessment – Questions 1 to 5

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?style=for-the-badge&logo=numpy&logoColor=white)

**Topics:** Sampling • Image Acquisition • Aliasing • Resolution • Computer Vision

</div>

---

## 📌 Requirements

Install the required libraries before running the programs:

```bash
pip install opencv-python numpy matplotlib
```

> **Note:** Keep an image named `input.jpg` in the same folder as the Python programs.

---

# 1. Effect of Sampling on Image Quality

### ❓ Question

A camera captures an image that appears blurred due to improper sampling. Explain how sampling affects image quality and suggest a method to correct it.

### 📝 Answer

**Sampling** is the process of converting a continuous image into a grid of pixels.

If the sampling rate is too low, fewer pixels are used to represent the image. As a result:

- Fine details may be lost.
- Edges may become unclear.
- The image may appear pixelated or blurred.
- Important visual information may not be represented correctly.

According to the **Nyquist Sampling Principle**, the sampling frequency should be sufficiently high to capture the details present in the original image.

### ✅ Corrective Method

Use a **higher sampling resolution** during image acquisition.

If only a low-resolution image is available, **interpolation techniques** such as Bicubic or Lanczos interpolation can be used while resizing to improve its visual appearance.

### 💻 Python Code

```python
import cv2
import matplotlib.pyplot as plt

# Read input image
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Simulate improper/low sampling
low_sampled = cv2.resize(
    img,
    (img.shape[1] // 5, img.shape[0] // 5),
    interpolation=cv2.INTER_AREA
)

# Restore size using bicubic interpolation
corrected = cv2.resize(
    low_sampled,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(low_sampled)
plt.title("Improper Sampling")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(corrected)
plt.title("Corrected Image")
plt.axis("off")

plt.tight_layout()
plt.show()
```

### 📤 Output

```text
Original Image
      ↓
Low Sampling Rate
      ↓
Loss of Fine Details / Pixelation
      ↓
Bicubic Interpolation
      ↓
Improved Visual Quality
```

**Observed Result:**  
The low-sampled image loses fine spatial details. Bicubic interpolation produces a smoother reconstructed image, although details permanently lost during sampling cannot be completely recovered.

---

# 2. Image Sensing and Acquisition Under Low Light

### ❓ Question

Given a real-world scene captured under low light, describe how the image sensing and acquisition process influences the final digital image.

### 📝 Answer

During **image acquisition**, light from a real-world scene enters the camera through the lens and reaches the image sensor.

The sensor converts incoming light into electrical signals, which are then converted into digital pixel values.

In low-light conditions:

- Fewer photons reach the sensor.
- The captured image becomes darker.
- Increasing sensor gain or ISO can introduce noise.
- Fine details and contrast may be reduced.
- Longer exposure can improve brightness but may cause motion blur.

Therefore, the quality of the final digital image depends greatly on the **sensor, exposure time, aperture, gain/ISO, and image-processing pipeline**.

### 💻 Python Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Simulate low-light acquisition
low_light = (img.astype(np.float32) * 0.25).astype(np.uint8)

# Simulate sensor noise
noise = np.random.normal(0, 12, low_light.shape)
noisy_low_light = np.clip(
    low_light.astype(np.float32) + noise,
    0,
    255
).astype(np.uint8)

# Brightness correction using gamma correction
gamma = 0.5

table = np.array([
    ((i / 255.0) ** gamma) * 255
    for i in np.arange(256)
]).astype("uint8")

enhanced = cv2.LUT(noisy_low_light, table)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Scene")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_low_light)
plt.title("Low-Light Acquisition")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(enhanced)
plt.title("Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()
```

### 📤 Output

```text
Real-World Scene
        ↓
Camera Lens
        ↓
Image Sensor
        ↓
Low Light → Weak Sensor Signal
        ↓
Noise + Reduced Detail
        ↓
Digital Image
        ↓
Gamma Correction
        ↓
Improved Brightness
```

**Observed Result:**  
The simulated low-light image becomes dark and noisy because of the reduced signal level. Gamma correction improves the visibility of darker regions, but sensor noise may still remain.

---

# 3. Aliasing – Sampling and Quantization

### ❓ Question

An image shows aliasing artifacts. Identify the cause using sampling and quantization concepts and propose a corrective approach.

### 📝 Answer

**Aliasing** occurs mainly when an image is sampled at a rate that is too low to represent high-frequency details.

It can produce:

- Jagged edges
- Moiré patterns
- False textures
- Distorted fine patterns

Sampling and quantization play different roles:

**Sampling:** Determines the number and position of pixels used to represent spatial image information.

**Quantization:** Determines the number of intensity or color levels available for each pixel.

Insufficient sampling mainly causes **spatial aliasing**, while insufficient quantization causes effects such as **banding or false contouring**.

### ✅ Corrective Approach

Aliasing can be reduced by:

1. Applying a **low-pass/anti-aliasing filter** before downsampling.
2. Increasing the sampling resolution.
3. Using appropriate interpolation methods.

### 💻 Python Code

```python
import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Improper downsampling
aliased = cv2.resize(
    img,
    (img.shape[1] // 4, img.shape[0] // 4),
    interpolation=cv2.INTER_NEAREST
)

# Apply anti-aliasing filter before sampling
blurred = cv2.GaussianBlur(img, (5, 5), 0)

anti_aliased = cv2.resize(
    blurred,
    (img.shape[1] // 4, img.shape[0] // 4),
    interpolation=cv2.INTER_AREA
)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(aliased)
plt.title("Aliased Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(anti_aliased)
plt.title("Anti-Aliased Image")
plt.axis("off")

plt.tight_layout()
plt.show()
```

### 📤 Output

```text
High-Frequency Image Details
          ↓
Insufficient Sampling
          ↓
Aliasing Artifacts
(Jagged Edges / Moiré Patterns)
          ↓
Low-Pass Filtering
          +
Proper Downsampling
          ↓
Reduced Aliasing
```

**Observed Result:**  
Direct low-resolution sampling can create jagged or distorted patterns. Applying a Gaussian low-pass filter before downsampling reduces high-frequency components and minimizes aliasing.

---

# 4. Pixel Resolution vs Intensity Resolution in Medical Imaging

### ❓ Question

Differentiate how pixel resolution and intensity resolution affect image representation in a practical application like medical imaging.

### 📝 Answer

Two important types of resolution in digital images are **pixel/spatial resolution** and **intensity resolution**.

| Feature | Pixel Resolution | Intensity Resolution |
|---|---|---|
| Meaning | Number of pixels representing the image | Number of intensity levels |
| Controls | Spatial detail | Brightness/contrast detail |
| Low Resolution Effect | Small structures may be lost | Subtle intensity differences may disappear |
| Example | 256×256 vs 1024×1024 | 4-bit vs 8-bit vs 12-bit |
| Medical Importance | Helps visualize small anatomical structures | Helps distinguish tissues with similar intensities |

In medical imaging, both are important.

For example, high spatial resolution can help visualize a small structure, while high intensity resolution can help distinguish tissues with very small differences in signal intensity.

### 💻 Python Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Reduce pixel/spatial resolution
low_pixel = cv2.resize(
    img,
    (128, 128),
    interpolation=cv2.INTER_AREA
)

low_pixel = cv2.resize(
    low_pixel,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# Reduce intensity resolution to 3 bits = 8 gray levels
levels = 8

low_intensity = np.floor(
    img / 256 * levels
) * (255 / (levels - 1))

low_intensity = np.clip(
    low_intensity,
    0,
    255
).astype(np.uint8)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(low_pixel, cmap="gray")
plt.title("Low Pixel Resolution")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(low_intensity, cmap="gray")
plt.title("Low Intensity Resolution")
plt.axis("off")

plt.tight_layout()
plt.show()
```

### 📤 Output

```text
Original Medical Image
        │
        ├───────────────┐
        ↓               ↓
Pixel Resolution    Intensity Resolution
        ↓               ↓
Spatial Detail      Gray-Level Detail
        ↓               ↓
Low Resolution      Low Resolution
        ↓               ↓
Loss of Small       Loss of Subtle
Structures          Tissue Differences
```

**Observed Result:**  
Reducing pixel resolution makes spatial structures appear blocky and less detailed. Reducing intensity resolution decreases the number of gray levels and can make subtle intensity differences harder to distinguish.

---

# 5. Computer Vision Level for Real-Time Surveillance

### ❓ Question

A surveillance system requires real-time image processing. Classify the level of Computer Vision involved and justify your choice.

### 📝 Answer

A complete intelligent surveillance system mainly involves **High-Level Computer Vision**, supported by low-level and mid-level processing.

### 🔹 Computer Vision Levels

| Level | Function | Surveillance Example |
|---|---|---|
| Low-Level Vision | Image enhancement and preprocessing | Noise removal, contrast enhancement |
| Mid-Level Vision | Feature extraction and object detection | Detecting people, vehicles or objects |
| High-Level Vision | Understanding and decision making | Recognizing events and generating alerts |

A real-time surveillance system that only enhances video is performing **low-level processing**.

However, a smart surveillance system that:

- Detects people or objects
- Tracks movement
- Interprets activities
- Makes decisions
- Generates alerts

is mainly a **high-level computer vision application**.

### 💻 Python Code

```python
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Background subtraction for motion detection
detector = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply motion detection
    mask = detector.apply(frame)

    # Remove small noise
    _, mask = cv2.threshold(
        mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # Find moving objects
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False

    for contour in contours:

        if cv2.contourArea(contour) > 1000:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            detected = True

    # Decision-making stage
    if detected:

        cv2.putText(
            frame,
            "MOVEMENT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Real-Time Surveillance", frame)
    cv2.imshow("Motion Mask", mask)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### 📤 Expected Output

```text
Camera / CCTV Video
        ↓
Image Acquisition
        ↓
Low-Level Vision
Preprocessing / Noise Reduction
        ↓
Mid-Level Vision
Motion / Object Detection
        ↓
High-Level Vision
Interpretation & Decision
        ↓
⚠ MOVEMENT DETECTED
```

When movement occurs, a bounding box is displayed around the moving region:

```text
┌──────────────────────────────────────────┐
│ REAL-TIME SURVEILLANCE                   │
│                                          │
│      ┌─────────────┐                     │
│      │   MOVING    │                     │
│      │   OBJECT    │                     │
│      └─────────────┘                     │
│                                          │
│          MOVEMENT DETECTED               │
└──────────────────────────────────────────┘
```

**Observed Result:**  
The system continuously processes camera frames, detects moving regions, and makes a simple decision by displaying an alert. A more advanced surveillance system can extend this pipeline with person detection, tracking, facial recognition, or activity recognition.

---

# 📊 Final Summary

| Q.No | Concept | Key Finding | Technique Used |
|---|---|---|---|
| **1** | Sampling | Low sampling causes loss of spatial detail | Bicubic interpolation |
| **2** | Image Acquisition | Low light reduces signal quality and increases noise | Gamma correction |
| **3** | Aliasing | Undersampling creates false spatial patterns | Anti-aliasing + proper sampling |
| **4** | Image Resolution | Pixel and intensity resolution affect different image details | Resizing + quantization |
| **5** | Computer Vision Levels | Intelligent surveillance mainly reaches high-level vision | Real-time motion detection |

---

## 🎯 Conclusion

This assessment demonstrates fundamental concepts of **Digital Image Processing and Computer Vision**.

The experiments show that:

- Proper **sampling** is necessary to preserve spatial image details.
- **Image sensing and acquisition** directly influence brightness, noise and overall image quality.
- **Aliasing** can be minimized using anti-aliasing filters and appropriate sampling.
- **Pixel resolution** controls spatial detail, while **intensity resolution** controls gray-level detail.
- A complete intelligent **surveillance system** combines low-, mid-, and high-level Computer Vision operations for real-time decision making.

---

<div align="center">

### 🛠️ Technologies Used

`Python` • `OpenCV` • `NumPy` • `Matplotlib`

### 📚 Digital Image Processing & Computer Vision Laboratory

**Assessment – Questions 1 to 5**

⭐ **End of Assessment** ⭐

</div>
