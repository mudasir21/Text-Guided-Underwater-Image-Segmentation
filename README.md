# Underwater Image Segmentation Web App

This web application allows users to upload underwater images and segment specific objects based on textual descriptions (referring expressions). It combines Grounding DINO for object detection and Segment Anything Model (SAM) for precise image segmentation.

## Features

- Drag-and-drop or clickable image upload
- User-defined referring expressions for segmentation (e.g., "yellow fish")
- Real-time preview of uploaded images
- Display of both original and segmented images
- Gallery view of all segmentation results
- Ability to download segmented images

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/underwater-image-segmentation.git
   cd underwater-image-segmentation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install flask torch torchvision opencv-python pillow numpy
   ```

4. Install Grounding DINO:
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   pip install -e .
   cd ..
   ```

5. Install Segment Anything Model:
   ```bash
   git clone https://github.com/facebookresearch/segment-anything.git
   cd segment-anything
   pip install -e .
   cd ..
   ```

6. Download the model weights:
   ```bash
   mkdir -p weights
   # Download Grounding DINO weights
   wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   
   # Download SAM weights
   wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

7. Set up the directory structure:
   ```bash
   mkdir -p static/uploads static/outputs templates
   ```

8. Add the application files to the project directory.

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload an underwater image by dragging and dropping or clicking on the upload area.

4. Enter a referring expression that describes what you want to segment (e.g., "yellow fish", "coral reef").

5. Click "Segment Image" and wait for the results to appear.

6. View the original and segmented images side by side.

7. Download the segmented image or view all previous results in the "All Results" page.

## Model Information

- **Grounding DINO**: Used for object detection based on textual descriptions.
- **Segment Anything Model (SAM)**: Used for precise image segmentation based on detected objects.

## Folder Structure

```
underwater-image-segmentation/
│
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── index.html      # Homepage with upload form
│   └── results.html    # Gallery of all results
│
├── static/             # Static files
│   ├── uploads/        # Stores uploaded images
│   └── outputs/        # Stores segmentation results
│
├── groundingdino/      # Grounding DINO repository
├── segment-anything/   # SAM repository
└── weights/            # Model weights
```

## Customization

- Adjust the detection thresholds in the `detect_objects` function to control sensitivity.
- Modify the visualization in the `visualize_segmentation` function to change the appearance of the segmented areas.
- Edit the HTML templates to customize the user interface.

## Troubleshooting

- If you encounter memory issues, try reducing the batch size or image resolution.
- Ensure your GPU has enough VRAM to handle the models (at least 8GB recommended).
- Check the console output for error messages if the segmentation fails.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for object detection
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything) for image segmentation