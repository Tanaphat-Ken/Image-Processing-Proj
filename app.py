from flask import Flask, render_template, request
import os
import cv2
from werkzeug.utils import secure_filename
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import matplotlib.cm as cm

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
CHECKPOINT_PATH = './checkpoints/kfold_2/best_epoch10.pth'
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

# Global storage for feature maps and gradients
feature_maps = {}
gradients = {}

def hook_feature_map(name):
    """Hook to capture feature maps."""
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

def hook_gradient(name):
    """Hook to capture gradients for Grad-CAM."""
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

def register_hooks(model):
    """Register forward and backward hooks on backbone layers."""
    hooks = []
    # Register hooks on ResNet backbone layers
    hooks.append(model.backbone.body.layer1.register_forward_hook(hook_feature_map('layer1')))
    hooks.append(model.backbone.body.layer2.register_forward_hook(hook_feature_map('layer2')))
    hooks.append(model.backbone.body.layer3.register_forward_hook(hook_feature_map('layer3')))
    hooks.append(model.backbone.body.layer4.register_forward_hook(hook_feature_map('layer4')))
    
    # Register backward hooks for Grad-CAM
    hooks.append(model.backbone.body.layer1.register_full_backward_hook(hook_gradient('layer1')))
    hooks.append(model.backbone.body.layer2.register_full_backward_hook(hook_gradient('layer2')))
    hooks.append(model.backbone.body.layer3.register_full_backward_hook(hook_gradient('layer3')))
    hooks.append(model.backbone.body.layer4.register_full_backward_hook(hook_gradient('layer4')))
    
    return hooks

# Register hooks on model
hooks = register_hooks(model)

def predict_image(image_path, model, device, score_threshold=0.3, compute_gradcam=False):
    """Run inference on an image and return predictions."""
    global feature_maps, gradients
    
    # Clear previous feature maps and gradients
    feature_maps.clear()
    gradients.clear()
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)
    
    if compute_gradcam:
        # Enable gradients for Grad-CAM
        image_tensor.requires_grad = True
        model.eval()
        predictions = model([image_tensor])[0]
        
        # Compute gradients with respect to the highest scoring detection
        if len(predictions['scores']) > 0:
            # Use the max score to compute gradients
            max_score = predictions['scores'].max()
            model.zero_grad()
            max_score.backward(retain_graph=True)
    else:
        # Run normal inference
        with torch.no_grad():
            predictions = model([image_tensor])[0]
    
    # Filter predictions by score threshold (use detach() when tensors require grad)
    scores_tensor = predictions['scores'].detach() if predictions['scores'].requires_grad else predictions['scores']
    keep = scores_tensor > score_threshold
    boxes = predictions['boxes'][keep].detach().cpu().numpy()
    scores = scores_tensor[keep].cpu().numpy()
    labels = predictions['labels'][keep].detach().cpu().numpy()
    
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

def visualize_feature_maps(layer_names=['layer1', 'layer2', 'layer3'], output_path=None, max_channels=16):
    """Visualize feature maps from specified layers."""
    global feature_maps
    
    num_layers = len(layer_names)
    fig, axes = plt.subplots(num_layers, max_channels, figsize=(max_channels * 2, num_layers * 2))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer_name in enumerate(layer_names):
        if layer_name not in feature_maps:
            continue
        
        fmap = feature_maps[layer_name][0]  # Get first image in batch
        num_channels = min(fmap.shape[0], max_channels)
        
        for j in range(num_channels):
            ax = axes[i, j]
            channel_map = fmap[j].cpu().numpy()
            ax.imshow(channel_map, cmap='viridis')
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(layer_name, fontsize=12, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

def compute_gradcam(layer_name='layer3', original_image=None):
    """Compute Grad-CAM heatmap for a specific layer."""
    global feature_maps, gradients
    
    if layer_name not in feature_maps or layer_name not in gradients:
        return None
    
    # Get feature maps and gradients
    fmap = feature_maps[layer_name][0]  # [C, H, W]
    grad = gradients[layer_name][0]     # [C, H, W]
    
    # Global average pooling on gradients
    weights = grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
    
    # Weighted combination of feature maps
    cam = (weights * fmap).sum(dim=0)  # [H, W]
    
    # Apply ReLU
    cam = F.relu(cam)
    
    # Normalize to [0, 1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam.cpu().numpy()

def visualize_gradcam_overlay(original_image, layer_names=['layer1', 'layer2', 'layer3'], output_path=None):
    """Create Grad-CAM overlay visualization for multiple layers."""
    num_layers = len(layer_names)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=((num_layers + 1) * 5, 5))
    
    # Show original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show Grad-CAM for each layer
    for i, layer_name in enumerate(layer_names):
        cam = compute_gradcam(layer_name, original_image)
        
        if cam is not None:
            # Resize CAM to match original image size
            img_size = original_image.size[::-1]  # (height, width)
            cam_resized = cv2.resize(cam, (img_size[1], img_size[0]))
            
            # Create heatmap
            heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
            
            # Overlay on original image
            img_array = np.array(original_image).astype(np.float32) / 255.0
            overlay = 0.5 * img_array + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)
            
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f'Grad-CAM: {layer_name}')
        else:
            axes[i + 1].text(0.5, 0.5, 'No gradient data', ha='center', va='center')
        
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
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

            # Run prediction with Grad-CAM computation
            boxes, scores, labels, image = predict_image(filepath, model, device, score_threshold=0.3, compute_gradcam=True)
            
            # Count detections (all are person class)
            count = len(boxes)
            
            # Draw and save output
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            
            out_filename = f"frcnn_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
            draw_predictions(image, boxes, scores, labels, output_path)
            
            # Generate feature maps visualization
            fmap_filename = f"fmap_{base_name}{ext}"
            fmap_path = os.path.join(app.config['OUTPUT_FOLDER'], fmap_filename)
            visualize_feature_maps(['layer1', 'layer2', 'layer3'], output_path=fmap_path, max_channels=8)
            
            # Generate Grad-CAM visualization
            gradcam_filename = f"gradcam_{base_name}{ext}"
            gradcam_path = os.path.join(app.config['OUTPUT_FOLDER'], gradcam_filename)
            visualize_gradcam_overlay(image, ['layer1', 'layer2', 'layer3'], output_path=gradcam_path)
            
            # Store results
            output_rel = os.path.join('outputs', out_filename).replace('\\', '/')
            fmap_rel = os.path.join('outputs', fmap_filename).replace('\\', '/')
            gradcam_rel = os.path.join('outputs', gradcam_filename).replace('\\', '/')
            
            results_data.append({
                'filename': file.filename,
                'model_name': 'Faster R-CNN (K-Fold 2)',
                'output_image': output_rel,
                'feature_maps': fmap_rel,
                'gradcam': gradcam_rel,
                'count': count,
                'avg_confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0
            })

    return render_template('index.html', results=results_data)


if __name__ == '__main__':
    app.run(debug=True)
