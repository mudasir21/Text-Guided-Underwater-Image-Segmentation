#                 updated codebase1.1
from flask import Flask, request, render_template, jsonify, url_for, redirect, send_from_directory
import os
import uuid
import torch
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for models
dino_model = None
sam_predictor = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_models():
    global dino_model, sam_predictor, device
    
    # Choose device - use CPU if CUDA is causing issues
    # If you're having problems with CUDA, you can force CPU usage by uncommenting the next line:
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Grounding DINO model
    dino_config = "/home/ghulam/mudasir/dlcv/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_checkpoint = "/home/ghulam/mudasir/dlcv/weights/groundingdino_swint_ogc.pth"
    dino_model = load_model(dino_config, dino_checkpoint)
    dino_model.to(device)
    
    # Load SAM model - make sure it's on the same device
    sam_checkpoint = "/home/ghulam/mudasir/dlcv/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    
    # Create predictor after moving model to device
    sam_predictor = SamPredictor(sam)
    
    # Explicitly set both models to evaluation mode
    dino_model.eval()
    sam.eval()
    
    print("Models loaded successfully")

# Function to load image from a local path
def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np = cv2.convertScaleAbs(image_np, alpha=1.2, beta=10)  # Enhance contrast
        image = Image.fromarray(image_np)
        return image
    except IOError as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

# Function to perform detection with Grounding DINO
def detect_objects(image, referring_expression, box_threshold=0.35, text_threshold=0.25):
    image_np = np.array(image)
    # converting rgb to bgr and then to pytorch tensor
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_tensor = torch.from_numpy(image_bgr).permute(2,0,1).float()/255.0
    
    # Move image tensor to the same device as the model
    image_tensor = image_tensor.to(device)
    
    # Perform detection
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=referring_expression,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # Ensure boxes are detached from the computation graph for easier handling
    boxes = boxes.detach()
    
    print(f"Detected {len(boxes)} objects for '{referring_expression}': {phrases}, Logits: {logits}")
    return boxes, image_np

# Function to perform segmentation with SAM
def segment_objects(image_np, boxes):
    sam_predictor.set_image(image_np)
    h, w = image_np.shape[:2]
    
    # Make sure boxes are on CPU before multiplication
    boxes_cpu = boxes.cpu()
    boxes_cpu = boxes_cpu * torch.tensor([w, h, w, h], dtype=torch.float32)
    
    masks = []
    for i, box in enumerate(boxes_cpu):
        # Convert to (x_min, y_min, x_max, y_max)
        x_center, y_center, box_w, box_h = box.numpy()
        x_min = x_center - box_w / 2
        x_max = x_center + box_w / 2
        y_min = y_center - box_h / 2
        y_max = y_center + box_h / 2
        
        # Validate box
        if box_w <= 0 or box_h <= 0 or x_min < 0 or x_max > w or y_min < 0 or y_max > h:
            print(f"Skipping invalid box {i}: x_min={x_min:.2f}, y_min={y_min:.2f}, x_max={x_max:.2f}, y_max={y_max:.2f}")
            continue
            
        # Ensure box coordinates are integers for SAM
        box_array = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        mask, _, _ = sam_predictor.predict(
            box=box_array,
            multimask_output=False
        )
        mask = mask[0]  # Get the first mask
        print(f"Mask {i} for box {box_array}: {mask.sum()} pixels")
        if mask.sum() > 0:  # Only include non-empty masks
            masks.append(mask)
            
    # Combine masks
    if not masks:
        print("No valid masks generated")
        return np.zeros((h, w), dtype=np.uint8)
        
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
    print(f"Combined mask: {combined_mask.sum()} pixels")
    return combined_mask

# Function to visualize segmentation
def visualize_segmentation(image, mask, output_path):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mask = cv2.resize(mask.astype(np.uint8), (image_np.shape[1], image_np.shape[0]))
    colored_mask = np.zeros_like(image_np)
    colored_mask[mask == 1] = [0, 0, 255]  # changed to red in BGR
    output = cv2.addWeighted(image_np, 0.7, colored_mask, 0.9, 0)
    cv2.imwrite(output_path, output)
    return output

# Main function to perform segmentation
def segment_image(image_path, referring_expression, output_path):
    try:
        # Load the image
        image = load_image(image_path)
        
        # Detect objects
        boxes, image_np = detect_objects(image, referring_expression)
        
        if len(boxes) == 0:
            print(f"No objects detected for: {referring_expression}")
            return None
        
        # Debug information about device
        if isinstance(boxes, torch.Tensor):
            print(f"Boxes device: {boxes.device}")
        
        # Segment objects
        mask = segment_objects(image_np, boxes)
        
        # Visualize and save the segmentation
        output = visualize_segmentation(image, mask, output_path)
        return output_path
        
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    referring_expression = request.form.get('expression', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Generate unique filenames
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        output_filename = str(uuid.uuid4()) + '.png'
        
        # Save paths
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save the uploaded file
        file.save(input_path)
        
        # Process the image
        try:
            result_path = segment_image(input_path, referring_expression, output_path)
            if result_path:
                return jsonify({
                    'success': True,
                    'original': '/static/uploads/' + filename,
                    'result': '/static/outputs/' + output_filename,
                    'expression': referring_expression
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f"No objects detected for: {referring_expression}"
                })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/results')
def results():
    # Get all files in output directory
    output_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) 
                    if os.path.isfile(os.path.join(app.config['OUTPUT_FOLDER'], f))]
    results = []
    for filename in output_files:
        results.append({
            'filename': filename,
            'path': '/static/outputs/' + filename
        })
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    # Set this to True to force CPU usage if you're having GPU issues
    FORCE_CPU = False
    
    if FORCE_CPU:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Forcing CPU usage for all operations")
    
    # Make sure PyTorch is using the right device
    if torch.cuda.is_available() and not FORCE_CPU:
        print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available or disabled, using CPU")
    
    initialize_models()
    app.run(debug=True)





















# from flask import Flask, request, render_template, jsonify, url_for, redirect, send_from_directory
# import os
# import uuid
# import torch
# from groundingdino.util.inference import load_model, predict
# from segment_anything import SamPredictor, sam_model_registry
# import cv2
# import numpy as np
# from PIL import Image
# import base64

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['OUTPUT_FOLDER'] = 'static/outputs'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# # Create folders if they don't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# # Global variables for models
# dino_model = None
# sam_predictor = None
# device = None

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def initialize_models():
#     global dino_model, sam_predictor, device
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Load Grounding DINO model
#     # dino_config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     dino_config = "/home/ghulam/mudasir/dlcv/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     dino_checkpoint = "/home/ghulam/mudasir/dlcv/weights/groundingdino_swint_ogc.pth"
#     dino_model = load_model(dino_config, dino_checkpoint)
#     dino_model.to(device)
    
#     # Load SAM model
#     # sam_checkpoint = "/home/ghulam/mudasir/dlcv/weights/sam_vit_h_4b8939.pth"
#     sam_checkpoint = "/home/ghulam/mudasir/dlcv/sam_vit_h_4b8939.pth"
#     sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
#     sam.to(device)
#     sam_predictor = SamPredictor(sam)
    
#     print("Models loaded successfully")

# # Function to load image from a local path
# def load_image(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         image_np = np.array(image)
#         image_np = cv2.convertScaleAbs(image_np, alpha=1.2, beta=10)  # Enhance contrast
#         image = Image.fromarray(image_np)
#         return image
#     except IOError as e:
#         raise RuntimeError(f"Failed to load image from {image_path}: {e}")

# # Function to perform detection with Grounding DINO
# def detect_objects(image, referring_expression, box_threshold=0.35, text_threshold=0.25):
#     image_np = np.array(image)
#     # converting rgb to bgr and then to pytorch tensor
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     image_tensor = torch.from_numpy(image_bgr).permute(2,0,1).float()/255.0
    
#     image_tensor = image_tensor.to(device)
    
#     # Perform detection
#     boxes, logits, phrases = predict(
#         model=dino_model,
#         image=image_tensor,
#         caption=referring_expression,
#         box_threshold=box_threshold,
#         text_threshold=text_threshold
#     )
#     print(f"Detected {len(boxes)} objects for '{referring_expression}': {phrases}, Logits: {logits}")
#     return boxes, image_np

# # Function to perform segmentation with SAM
# def segment_objects(image_np, boxes):
#     sam_predictor.set_image(image_np)
#     h, w = image_np.shape[:2]
#     # Convert boxes from normalized (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
#     boxes = boxes * torch.tensor([w, h, w, h], device=device)
#     masks = []
#     for i, box in enumerate(boxes):
#         # Convert to (x_min, y_min, x_max, y_max)
#         x_center, y_center, box_w, box_h = box.cpu().numpy()
#         x_min = x_center - box_w / 2
#         x_max = x_center + box_w / 2
#         y_min = y_center - box_h / 2
#         y_max = y_center + box_h / 2
#         # Validate box
#         if box_w <= 0 or box_h <= 0 or x_min < 0 or x_max > w or y_min < 0 or y_max > h:
#             print(f"Skipping invalid box {i}: x_min={x_min:.2f}, y_min={y_min:.2f}, x_max={x_max:.2f}, y_max={y_max:.2f}")
#             continue
#         # Ensure box coordinates are integers for SAM
#         box_array = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
#         mask, _, _ = sam_predictor.predict(
#             box=box_array,
#             multimask_output=False
#         )
#         mask = mask[0]  # Get the first mask
#         print(f"Mask {i} for box {box_array}: {mask.sum()} pixels")
#         if mask.sum() > 0:  # Only include non-empty masks
#             masks.append(mask)
#     # Combine masks
#     if not masks:
#         print("No valid masks generated")
#         return np.zeros((h, w), dtype=np.uint8)
#     combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
#     for mask in masks:
#         combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
#     print(f"Combined mask: {combined_mask.sum()} pixels")
#     return combined_mask

# # Function to visualize segmentation
# def visualize_segmentation(image, mask, output_path):
#     image_np = np.array(image)
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     mask = cv2.resize(mask.astype(np.uint8), (image_np.shape[1], image_np.shape[0]))
#     colored_mask = np.zeros_like(image_np)
#     colored_mask[mask == 1] = [255, 0, 0]  # changed to red in BGR
#     output = cv2.addWeighted(image_np, 0.7, colored_mask, 0.8, 0)
#     cv2.imwrite(output_path, output)
#     return output

# # Main function to perform segmentation
# def segment_image(image_path, referring_expression, output_path):
#     image = load_image(image_path)
#     boxes, image_np = detect_objects(image, referring_expression)
#     if len(boxes) == 0:
#         print(f"No objects detected for: {referring_expression}")
#         return None
#     mask = segment_objects(image_np, boxes)
#     output = visualize_segmentation(image, mask, output_path)
#     return output_path

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     referring_expression = request.form.get('expression', '')
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     if file and allowed_file(file.filename):
#         # Generate unique filenames
#         filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
#         output_filename = str(uuid.uuid4()) + '.png'
        
#         # Save paths
#         input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
#         # Save the uploaded file
#         file.save(input_path)
        
#         # Process the image
#         try:
#             result_path = segment_image(input_path, referring_expression, output_path)
#             if result_path:
#                 return jsonify({
#                     'success': True,
#                     'original': '/static/uploads/' + filename,
#                     'result': '/static/outputs/' + output_filename,
#                     'expression': referring_expression
#                 })
#             else:
#                 return jsonify({
#                     'success': False,
#                     'error': f"No objects detected for: {referring_expression}"
#                 })
#         except Exception as e:
#             return jsonify({'error': str(e)})
    
#     return jsonify({'error': 'Invalid file type'})

# @app.route('/results')
# def results():
#     # Get all files in output directory
#     output_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) 
#                     if os.path.isfile(os.path.join(app.config['OUTPUT_FOLDER'], f))]
#     results = []
#     for filename in output_files:
#         results.append({
#             'filename': filename,
#             'path': '/static/outputs/' + filename
#         })
    
#     return render_template('results.html', results=results)

# if __name__ == '__main__':
#     initialize_models()
#     app.run(debug=True)