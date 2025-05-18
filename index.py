import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import List, Tuple, Dict
import threading
from allDataCopy import start


start("20200226-GTI-02.webp")


# ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")  

class CarAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Car Detection and Analysis System")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        self.image_path = ""
        self.original_image = None
        self.detected_cars = []  
        self.car_colors = []     
        self.plate_texts = []    
        self.current_car_index = 0
        
        
        self.create_ui()
        
    def create_ui(self):
        """Create the main UI elements"""
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        
        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.create_control_panel()
        
        
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.create_display_panel()
        
        
        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
    def create_control_panel(self):
        """Create the control panel elements"""
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        
        title_label = ctk.CTkLabel(self.left_panel, text="Car Detection and Analysis", 
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        
        img_frame = ctk.CTkFrame(self.left_panel)
        img_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        img_label = ctk.CTkLabel(img_frame, text="Input Image:")
        img_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.image_path_label = ctk.CTkLabel(img_frame, text="No image selected", 
                                            fg_color=("gray90", "gray20"), corner_radius=5)
        self.image_path_label.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        browse_btn = ctk.CTkButton(img_frame, text="Browse", command=self.browse_image)
        browse_btn.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        
        process_btn = ctk.CTkButton(self.left_panel, text="Detect Cars", 
                                  command=self.process_image, fg_color="green", hover_color="dark green")
        process_btn.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        
        
        nav_frame = ctk.CTkFrame(self.left_panel)
        nav_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        nav_label = ctk.CTkLabel(nav_frame, text="Car Navigation:")
        nav_label.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        prev_btn = ctk.CTkButton(nav_frame, text="Previous", command=self.previous_car, width=80)
        prev_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.car_counter_label = ctk.CTkLabel(nav_frame, text="0/0")
        self.car_counter_label.grid(row=1, column=1, padx=10, pady=5)
        
        next_btn = ctk.CTkButton(nav_frame, text="Next", command=self.next_car, width=80)
        next_btn.grid(row=1, column=2, padx=5, pady=5)
        
        
        results_frame = ctk.CTkFrame(self.left_panel)
        results_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        results_label = ctk.CTkLabel(results_frame, text="Current Car Details:")
        results_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.car_color_label = ctk.CTkLabel(results_frame, text="Color: None")
        self.car_color_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.plate_text_label = ctk.CTkLabel(results_frame, text="License Plate: None")
        self.plate_text_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        
        save_btn = ctk.CTkButton(self.left_panel, text="Save Results", command=self.save_results)
        save_btn.grid(row=5, column=0, padx=20, pady=20, sticky="ew")
        
        
        self.progress_bar = ctk.CTkProgressBar(self.left_panel)
        self.progress_bar.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)
        
        
        spacer = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        spacer.grid(row=7, column=0, sticky="ew", pady=10)
        self.left_panel.grid_rowconfigure(7, weight=1)
        
    def create_display_panel(self):
        """Create the display panel elements"""
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)
        
        
        self.original_image_frame = ctk.CTkFrame(self.right_panel)
        self.original_image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        orig_label = ctk.CTkLabel(self.original_image_frame, text="Original Image:")
        orig_label.pack(anchor="w", padx=10, pady=5)
        
        self.original_image_canvas = tk.Canvas(self.original_image_frame, bg="red")
        self.original_image_canvas.pack(fill="both", expand=True, padx=10, pady=5)
        
        
        self.car_image_frame = ctk.CTkFrame(self.right_panel)
        self.car_image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.car_label = ctk.CTkLabel(self.car_image_frame, text="Detected Car:")
        self.car_label.pack(anchor="w", padx=10, pady=5)
        
        self.car_image_canvas = tk.Canvas(self.car_image_frame, bg="white")
        self.car_image_canvas.pack(fill="both", expand=True, padx=10, pady=5)
    
    def browse_image(self):
        """Open file dialog to select an image"""
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        )
        
        filepath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes
        )
        
        if filepath:
            self.image_path = filepath
            self.image_path_label.configure(text=os.path.basename(filepath))
            self.load_original_image(filepath)
            self.reset_detection_results()
            
    def load_original_image(self, filepath):
        """Load and display the original image"""
        try:
            
            self.original_image = Image.open(filepath)
            
            
            self.display_image_on_canvas(self.original_image, self.original_image_canvas)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image_on_canvas(self, image, canvas):
        """Display an image on the specified canvas"""
        if image is None:
            return
            
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 300
        
        
        img_width, img_height = image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        
        
        photo_img = ImageTk.PhotoImage(resized_img)
        
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo_img, anchor="center")
        
        
        canvas.image = photo_img
    
    def process_image(self):
        """Process the image to detect cars and extract information"""
        if not self.original_image or not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        
        self.status_bar.configure(text="Processing image...")
        self.progress_bar.set(0.1)
        
        
        threading.Thread(target=self._run_detection, daemon=True).start()
    
    def _run_detection(self):
        """Run the car detection and analysis in a separate thread"""
        try:
            
            open_cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            
            
            self.after(100, lambda: self.progress_bar.set(0.2))
            self.after(100, lambda: self.status_bar.configure(text="Detecting cars..."))
            
            
            
            detected_cars, car_colors = self.simulate_car_detection(open_cv_image)
            
            
            self.after(100, lambda: self.progress_bar.set(0.6))
            self.after(100, lambda: self.status_bar.configure(text="Extracting license plates..."))
            
            
            plate_texts = []
            for car_img in detected_cars:
                
                plate_text = self.simulate_license_plate_extraction(car_img)
                plate_texts.append(plate_text)
            
            
            self.after(100, lambda: self.update_with_detection_results(detected_cars, car_colors, plate_texts))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.after(0, lambda: self.status_bar.configure(text="Error during processing"))
            self.after(0, lambda: self.progress_bar.set(0))
    
    def update_with_detection_results(self, detected_cars, car_colors, plate_texts):
        """Update the UI with detection results"""
        self.detected_cars = detected_cars
        self.car_colors = car_colors
        self.plate_texts = plate_texts
        self.current_car_index = 0
        
        
        self.car_counter_label.configure(text=f"1/{len(detected_cars)}" if detected_cars else "0/0")
        
        
        if detected_cars:
            self.display_current_car()
        
        
        self.progress_bar.set(1.0)
        self.status_bar.configure(text=f"Completed. Found {len(detected_cars)} cars.")
        
        
        self.after(1000, lambda: self.progress_bar.set(0))
    
    def display_current_car(self):
        """Display the current car and its details"""
        if not self.detected_cars or self.current_car_index >= len(self.detected_cars):
            return
        
        
        car_img = self.detected_cars[self.current_car_index]
        car_color = self.car_colors[self.current_car_index]
        plate_text = self.plate_texts[self.current_car_index]
        
        car_pil_img = Image.fromarray(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB))
        
        self.display_image_on_canvas(car_pil_img, self.car_image_canvas)
        
        self.car_color_label.configure(text=f"Color: {car_color}")
        self.plate_text_label.configure(text=f"License Plate: {plate_text}")
        
        self.car_counter_label.configure(text=f"{self.current_car_index + 1}/{len(self.detected_cars)}")
    
    def next_car(self):
        """Navigate to the next car"""
        if not self.detected_cars:
            return
            
        self.current_car_index = (self.current_car_index + 1) % len(self.detected_cars)
        self.display_current_car()
    
    def previous_car(self):
        """Navigate to the previous car"""
        if not self.detected_cars:
            return
            
        self.current_car_index = (self.current_car_index - 1) % len(self.detected_cars)
        self.display_current_car()
    
    def save_results(self):
        """Save the detection results to a file"""
        if not self.detected_cars:
            messagebox.showwarning("Warning", "No detection results to save.")
            return
        
        try:
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Results As"
            )
            
            if not file_path:
                return
                
            
            with open(file_path, 'w') as f:
                f.write(f"Car Detection Results - {len(self.detected_cars)} cars found\n")
                f.write(f"Source Image: {self.image_path}\n\n")
                
                for i, (color, plate) in enumerate(zip(self.car_colors, self.plate_texts)):
                    f.write(f"Car {i + 1}:\n")
                    f.write(f"  Color: {color}\n")
                    f.write(f"  License Plate: {plate}\n\n")
            
            messagebox.showinfo("Success", f"Results saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def reset_detection_results(self):
        """Reset all detection results"""
        self.detected_cars = []
        self.car_colors = []
        self.plate_texts = []
        self.current_car_index = 0
        
        
        self.car_counter_label.configure(text="0/0")
        self.car_color_label.configure(text="Color: None")
        self.plate_text_label.configure(text="License Plate: None")
        
        
        self.car_image_canvas.delete("all")
    
    
    def simulate_car_detection(self, image) -> Tuple[List[np.ndarray], List[str]]:
        """Simulate car detection - replace with your actual ML model function"""
        
        
        
        
        height, width = image.shape[:2]
        num_cars = np.random.randint(2, 4)
        
        detected_cars = []
        car_colors = []
        
        colors = ["Red", "Blue", "Black", "White", "Silver", "Gray"]
        
        for i in range(num_cars):
            
            crop_h = np.random.randint(height // 4, height // 2)
            crop_w = np.random.randint(width // 4, width // 2)
            
            x = np.random.randint(0, width - crop_w)
            y = np.random.randint(0, height - crop_h)
            
            car_crop = image[y:y+crop_h, x:x+crop_w].copy()
            
            
            cv2.rectangle(car_crop, (5, 5), (crop_w-5, crop_h-5), (0, 255, 0), 2)
            
            detected_cars.append(car_crop)
            car_colors.append(np.random.choice(colors))
        
        
        import time
        time.sleep(1)
        
        return detected_cars, car_colors
    
    def simulate_license_plate_extraction(self, car_image) -> str:
        """Simulate license plate text extraction - replace with your actual OCR function"""
        
        
        
        
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        numbers = "0123456789"
        
        plate_format = np.random.choice([
            "LL-NNN-LL",  
            "LL-NN-NNN",  
            "NNN-LL-NN"   
        ])
        
        plate_text = ""
        for char in plate_format:
            if char == "L":
                plate_text += np.random.choice(list(letters))
            elif char == "N":
                plate_text += np.random.choice(list(numbers))
            else:
                plate_text += char
        
        
        import time
        time.sleep(0.5)
        
        return plate_text


if __name__ == "__main__":
    app = CarAnalysisApp()
    app.mainloop()