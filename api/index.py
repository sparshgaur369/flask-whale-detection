import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO

app = Flask(__name__)

# Load the trained YOLO model
MODEL_PATH = "model/yolo_model.pt"
model = YOLO(MODEL_PATH)  # Directly load from .pt

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Function to process images and detect whales
def process_images(upload_folder):
    results_data = []
    outlier_threshold = 1.5  # Define threshold for outliers

    for img_name in os.listdir(upload_folder):
        img_path = os.path.join(upload_folder, img_name)
        
        # Ensure the image is valid
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading {img_name}, skipping...")
            continue

        # Run YOLO detection
        results = model(img_path)  # Use YOLO directly
        detections = len(results[0].boxes)  # Number of detected whales

        results_data.append({"Image": img_name, "Detections": detections})

    # Convert results to DataFrame
    df = pd.DataFrame(results_data)
    if df.empty:
        print("No valid detections found!")
        return None, None

    # Calculate mean and standard deviation
    mean_count = df["Detections"].mean()
    std_dev = df["Detections"].std()

    # Identify outliers
    df["Outlier"] = df["Detections"].apply(lambda x: abs(x - mean_count) > outlier_threshold * std_dev)

    # Save results to CSV
    csv_path = os.path.join(RESULTS_FOLDER, "whale_detection_results.csv")
    df.to_csv(csv_path, index=False)

    return df, csv_path

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Clear previous uploads
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, file))

        # Save uploaded images
        uploaded_files = request.files.getlist("images")
        for file in uploaded_files:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

        # Process images
        results_df, csv_path = process_images(UPLOAD_FOLDER)

        return render_template("index.html", results=results_df.to_dict(orient="records") if results_df is not None else [], csv_path=csv_path)

    return render_template("index.html", results=None)

@app.route("/download")
def download_csv():
    return send_file("results/whale_detection_results.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
