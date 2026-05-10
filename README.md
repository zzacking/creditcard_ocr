# Chinese version
# Credit Card OCR - 基于传统图像处理的信用卡数字识别

> 无需深度学习，纯 OpenCV 实现信用卡凸印数字识别。自适应定位算法，适应不同卡片布局。

## 特点

- **自适应定位**：通过寻找连续稳定宽度值定位数字区域，无需硬编码坐标
- **高度修正**：剔除异常值后取平均，自动修正粘连导致的轮廓高度异常
- **光照鲁棒**：Sobel 边缘检测 + 大津法（OTSU）自动阈值，适应不同光照条件
- **纯传统方法**：仅依赖 OpenCV，无需 Tesseract、无需深度学习框架

## 核心思路
原始图像 → 顶帽变换(去光照不均) → Sobel(提取边缘) → 大津法(二值化)
→ 闭运算(连接数字块) → 稳定值算法(定位ROI) → 高度修正
→ 轮廓提取(分割单字) → 模板匹配(识别数字)

## 环境依赖

- Python 3.x
- OpenCV >= 4.x（开发环境 4.13.0）
- NumPy >= 2.x（开发环境 2.2.6）

## 使用说明

1. 将模板图片 `number_ref.png` 放入 `img_collection/` 目录
2. 将信用卡图片命名为 `creditcard*.png` 放入同一目录
3. 运行脚本

```bash
python card_ocr.py
```

# English version
# Credit Card OCR - Embossed Digit Recognition via Traditional Image Processing

> No deep learning required. Pure OpenCV implementation for recognizing embossed credit card numbers. Adaptive localization adapts to different card layouts.

## Features

- **Adaptive Localization**: Finds digit regions by detecting consecutive stable widths — no hard-coded coordinates
- **Height Correction**: Removes outliers and uses mean height to fix contour粘连 caused height anomalies
- **Illumination Robust**: Sobel edge detection + Otsu's adaptive thresholding for varying lighting
- **Pure Traditional CV**: OpenCV only, no Tesseract, no deep learning frameworks

## Pipeline
Raw Image → Top-hat (illumination normalization) → Sobel (edge extraction) → Otsu (binarization)
→ Morph-close (merge digit blocks) → Stable-value algorithm (ROI location) → Height correction
→ Contour extraction (single digit segmentation) → Template matching (digit recognition)

## Requirements

- Python 3.x
- OpenCV >= 4.x (developed on 4.13.0)
- NumPy >= 2.x (developed on 2.2.6)

## Usage

1. Place template image `number_ref.png` in `img_collection/`
2. Place credit card images named `creditcard*.png` in the same folder
3. Run the script

```bash
python card_ocr.py
