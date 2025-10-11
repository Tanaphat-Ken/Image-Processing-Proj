from flask import Flask, render_template, request
import os
import cv2
from werkzeug.utils import secure_filename
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
CHECKPOINT_PATH = 'checkpoints/kfold_2/best_epoch10.pth'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Number of classes (including background)
NUM_CLASSES = 2  # background + person

# Load the Faster R-CNN model
def load_model(checkpoint_path, num_classes, device):
    """Load Faster R-CNN model with custom weights."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint info: Fold={checkpoint.get('fold')}, Epoch={checkpoint.get('epoch')}, Val Loss={checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model

# Initialize model
model = load_model(CHECKPOINT_PATH, NUM_CLASSES, device)

# Image transform (ImageNet normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model, device, score_threshold=0.3):
    """Run inference on an image and return predictions."""
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    # Filter predictions by score threshold
    keep = predictions['scores'] > score_threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    
    return boxes, scores, labels, image

def draw_predictions(image, boxes, scores, labels, output_path):
    """Draw bounding boxes on image and save."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw each box
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add score text
        ax.text(
            x1, y1 - 5,
            f'{score:.2f}',
            color='red',
            fontsize=10,
            backgroundcolor='white'
        )
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    results_data = None

    if request.method == 'POST':
        uploaded_files = request.files.getlist("files")
        if not uploaded_files:
            return render_template('index.html', results=None)

        results_data = []

        for file in uploaded_files[:5]:  # Limit to 5 images
            filename = secure_filename(file.filename)
            if filename == '':
                continue
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run prediction
            boxes, scores, labels, image = predict_image(filepath, model, device, score_threshold=0.3)
            
            # Count detections (all are person class)
            count = len(boxes)
            
            # Draw and save output
            out_filename = f"frcnn_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
            draw_predictions(image, boxes, scores, labels, output_path)
            
            # Store results
            output_rel = os.path.join('outputs', out_filename).replace('\\', '/')
            
            results_data.append({
                'filename': file.filename,
                'model_name': 'Faster R-CNN (K-Fold 2)',
                'output_image': output_rel,
                'count': count,
                'avg_confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0
            })

    return render_template('index.html', results=results_data)


if __name__ == '__main__':
    app.run(debug=True)
