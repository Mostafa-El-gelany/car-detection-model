# Car Detection Model

A desktop computer vision app that detects cars in an image, estimates the dominant color of each car, and extracts license plate text with OCR. The app uses a CustomTkinter GUI, YOLOv5 for vehicle detection, and EasyOCR for text recognition.

## Features

- Load an image from disk and preview it in the app
- Detect cars in the image using a pretrained YOLOv5 model
- Analyze the dominant color of each detected car
- Extract license plate text with OCR
- Browse through detected cars with Previous / Next controls
- Save the detection summary to a text file

## Project Structure

- `app.py` - GUI application entry point
- `src/allDataCopy.py` - Active detection pipeline used by the app
- `src/allData.py` - Alternate backend implementation
- `yolov5s.pt` - Pretrained YOLOv5 model file
- `images/` - Sample images and example outputs

## Requirements

Install Python 3.10+ and the following packages:

- `customtkinter`
- `Pillow`
- `opencv-python`
- `numpy`
- `torch`
- `easyocr`
- `matplotlib`

> On the first run, YOLOv5 and EasyOCR may download model files if they are not already cached.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install customtkinter Pillow opencv-python numpy torch easyocr matplotlib
```

If you already have a working Python environment, install only the missing packages there.

## Run

Start the application with:

```bash
python app.py
```

## How It Works

1. Click Browse and choose an image.
2. Click Detect Cars to run the detection pipeline.
3. Review the annotated source image and the current car crop.
4. Use Previous and Next to move through detections.
5. Click Save Results to export a text summary.

## Notes

- The app currently uses `src/allDataCopy.py` as the backend processor.
- Results depend on image quality, lighting, and plate visibility.
- OCR accuracy can vary significantly for blurred, angled, or low-resolution plates.

## Troubleshooting

- If the app fails to start, make sure the required packages are installed in the same Python environment you use to run `app.py`.
- If detection is slow on the first run, it is likely downloading model weights.
- If OCR returns poor results, try a clearer image with a larger, sharper plate area.

## Example Images

Sample images are available in the `images/` folder for quick testing.