import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize with English language

# Load pretrained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def load_image(image_path):
    """Load and validate image (returns BGR image)"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img  # BGR format

def get_dominant_color(crop, k=3):
    """Get dominant colors using k-means clustering (works with BGR input)"""
    if crop.size == 0:
        return (0, 0, 0), 0

    # Convert BGR to RGB for color analysis
    rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = rgb_img.reshape(-1, 3).astype(np.float32)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get the dominant cluster
    counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(counts)
    dominant_color = centers[dominant_idx].astype(int)

    # Calculate the percentage of dominant color
    dominance = counts[dominant_idx] / sum(counts)

    return tuple(dominant_color), dominance  # Returns RGB color

def rgb_to_color_name(rgb, dominance):
    """Improved color classification using RGB color space"""
    r, g, b = rgb
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    hue, sat, val = hsv

    # First check for black/white/gray
    if val < 40:
        return 'black'
    elif sat < 30 and val > 200:
        return 'white'
    elif sat < 50 and val > 170:# < 50    > 180
        return 'gray'



    # Then check for colors (with dominance threshold)
    if dominance < 0.3:  # Not strongly dominant color
        if val > 180:
            return 'light metallic'
        elif val > 120:
            return 'metallic'
        else:
            return 'dark metallic'



    # Strong dominant colors
    if hue < 10 or hue > 170:
        return 'red'
    elif 10 <= hue < 25:
        if r > 200 and g > 150:
            return 'orange'
        return 'gold' if r > 180 and g > 150 and b < 100 else 'orange'
    elif 25 <= hue < 35:
        return 'yellow'
    elif 35 <= hue < 85:
        return 'green'
    elif 85 <= hue < 125:
        return 'blue'
    elif 125 <= hue < 150:
        return 'purple'
    elif 150 <= hue <= 170:
        return 'pink'

    return 'unknown'

def process_ocr(car_crop):
    """Perform OCR on a car crop and return text from the largest rectangle (by area)"""
    # Convert BGR to RGB (EasyOCR expects RGB)
    rgb_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
    
    # Perform OCR and get all detections with bounding boxes
    results = reader.readtext(rgb_crop)
    
    if not results:
        return "No text detected"
    
    # Find the detection with the largest bounding box area
    largest_area = 0
    largest_text = "No text detected"
    largest_bbox = None
    
    for (bbox, text, prob) in results:
        if prob > 0.3:  # Only consider detections with >40% confidence
            # Calculate area of the bounding box
            (top_left, top_right, bottom_right, bottom_left) = bbox
            width = top_right[0] - top_left[0]
            height = bottom_left[1] - top_left[1]
            area = width * height
            
            # Update if this is the largest area so far
            if area > largest_area:
                largest_area = area
                largest_text = text
                largest_bbox = bbox
    
    # Draw the largest bounding box on the original image for visualization
    if largest_bbox is not None:
        (top_left, top_right, bottom_right, bottom_left) = largest_bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(car_crop, top_left, bottom_right, (0, 255, 255), 2)
    
    return largest_text

def process_image(img_path, conf_threshold=0.1):
    """Main processing pipeline (works with BGR image)"""
    try:
        img = load_image(img_path)  # BGR image
        car_crops = []  # To store individual car crops
        car_info = []  # To store information about each car

        # Run detection
        results = model(img)

        # Filter for cars with sufficient confidence
        car_detections = results.xyxy[0][(results.xyxy[0][:, 5] == 2) & (results.xyxy[0][:, 4] > conf_threshold)]

        if len(car_detections) == 0:
            print("No cars detected with sufficient confidence.")
            return None, [], []

        for i, (*box, conf, cls) in enumerate(car_detections):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            car_crop = img[y1:y2, x1:x2]  # BGR crop
            if car_crop.size == 0:
                continue

            # Store the cropped car image
            car_crops.append(car_crop)

            # Get dominant color
            dominant_rgb, dominance = get_dominant_color(car_crop)
            color = rgb_to_color_name(dominant_rgb, dominance)
            
            # Perform OCR on the car crop
            detected_text = process_ocr(car_crop)
            
            # Store car information
            car_info.append({
                'color': color,
                'text': detected_text,
                'coordinates': (x1, y1, x2, y2),
                'confidence': float(conf)
            })

            # Draw results on original BGR image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{color} car - {detected_text[:15]}'  # Show first 15 chars of detected text
            cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img, car_crops, car_info  # Returns annotated image, crops, and info

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, [], []

def display_results(main_img, car_crops, car_info):
    """Display the main image and individual car crops with OCR results"""
    # Display main image with detections using matplotlib
    print("\nMain Image with Detections:")
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB))
    plt.title('Main Image with Detections')
    plt.axis('off')
    plt.show()
    
    # Display each cropped car with OCR results
    if car_crops:
        print("\nIndividual Car Crops with OCR Results:")
        plt.figure(figsize=(15, 5))
        for i, (crop, info) in enumerate(zip(car_crops, car_info), 1):
            plt.subplot(1, len(car_crops), i)
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f'Car {i}\nColor: {info["color"]}\nText: {info["text"]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Change to your image path
    image_path = 'qvfr8eyzcvz31.jpg'

    # Process and display
    result_img, car_crops, car_info = process_image(image_path)  # Returns BGR image, crops, and info

    if result_img is not None:
        # Display results
        display_results(result_img, car_crops, car_info)

        # Save results
        try:
            
            cv2.imwrite('detected_cars.jpg', result_img)
            '''
            for i, crop in enumerate(car_crops, 1):
                cv2.imwrite(f'car_{i}.jpg', crop)'''
            
            
            print("\nResults saved:")
            print("- Main image: 'detected_cars.jpg'")
            print(f"- Individual cars: {len(car_crops)} files (car_1.jpg, etc.)")
            
            # Print detailed OCR results
            print("\nDetailed Car Information:")
            for i, info in enumerate(car_info, 1):
                print(f"\nCar {i}:")
                print(f"  Color: {info['color']}")
                print(f"  Detected text: {info['text']}")
                print(f"  Coordinates: {info['coordinates']}")
                print(f"  Detection confidence: {info['confidence']:.2f}")
        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("No cars detected or error occurred.")
