# LCSOD Detection Tool
A Streamlit-based web application for computer vision object detection using state-of-the-art models.

## Features
- **Single Image Detection**: Upload and analyze individual images
- **Multiple Model Support**: Choose from ResNet-50, FocalNet-Large, and Swin-Large
- **Real-time Visualization**: Interactive detection results with confidence scores
- **Export Functionality**: Save detection results and visualizations

## Installation
Use the provided launcher scripts:

**Windows:**
```bash
start_app.bat
```
**Linux/Mac:**
```bash
start_app.sh
```

![image]([picture or gif url)](https://github.com/xyn-abc/LCSOD-tool/blob/main/LCSOD%20tool%20guidance.gif)

### Using the Interface
1. **Navigate to Object Detection**: Select the "Object Detection" page from the sidebar
2. **Upload Image**: Use the file uploader to select an image (JPG, PNG, BMP, TIFF)
3. **Configure Settings**:
   - Choose a model (ResNet-50, FocalNet-Large, or Swin-Large)
   - Select dataset configuration (Data1 or Data2)
   - Adjust confidence threshold (0.1 - 0.9)
4. **Run Detection**: Click "Run Detection" to analyze the image
5. **View Results**: Explore detection results in multiple tabs:
   - **Detection Results**: Annotated image with bounding boxes
   - **Original Image**: Unprocessed input image
   - **Detection Details**: Table with detection information and charts

### Export Options
- **Export Detection Data**: Download detection results as JSON
- **Save Result Image**: Download the annotated image as PNG


### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for large models)
- **Storage**: At least 5GB free space for models

### Software
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **Operating System**: Windows 10+, Linux

### Performance Tips
- Use GPU acceleration for faster inference
- Choose ResNet-50 for fastest inference speed
- Use FocalNet-Large or Swin-Large for highest accuracy
- Adjust confidence threshold to filter results

### Adding New Models
1. Add model configuration file to `/configs/`
2. Add model weights to `/model/`

## License
This project is licensed under the MIT License. See LICENSE file for details.


