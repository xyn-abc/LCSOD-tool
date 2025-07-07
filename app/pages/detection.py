import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import json
import sys
import os
from datetime import datetime
import tempfile
import io
import logging
from pathlib import Path
from typing import Dict

# Add backend to path - more robust path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import main_test functions
from backend.util.lazy_load import Config
from backend.util.utils import load_checkpoint, load_state_dict
from backend.datasets.coco import CocoDetection
from backend.util.collate_fn import collate_fn
from backend.util.misc import fixed_generator, seed_worker
from accelerate import Accelerator
from torch.utils import data
from pycocotools.coco import COCO
from backend.util.coco_eval import filter_bbox, filter_bbox2
from backend.util.visualize import plot_bounding_boxes_on_image

def get_available_models():
    """Get list of available models"""
    models = [
        "ResNet-50 (Data1)",
        "FocalNet-Large (Data1)", 
        "Swin-Large (Data1)",
        "ResNet-50 (Data2)",
        "FocalNet-Large (Data2)",
        "Swin-Large (Data2)"
    ]
    return models

def get_available_datasets():
    """Get list of available datasets"""
    datasets = ["data1", "data2"]
    return datasets

def get_model_config_mapping():
    """Get mapping from model names to config files"""
    return {
        "ResNet-50 (Data1)": "configs/resnet_data1.py",
        "FocalNet-Large (Data1)": "configs/focal_data1.py",
        "Swin-Large (Data1)": "configs/swin_data1.py",
        "ResNet-50 (Data2)": "configs/resnet_data2.py", 
        "FocalNet-Large (Data2)": "configs/focal_data2.py",
        "Swin-Large (Data2)": "configs/swin_data2.py"
    }

def get_model_checkpoint_mapping():
    """Get mapping from model names to checkpoint files"""
    return {
        "ResNet-50 (Data1)": "model/resnet_data1.pth",
        "FocalNet-Large (Data1)": "model/focal_data1.pth",
        "Swin-Large (Data1)": "model/swin_data1.pth",
        "ResNet-50 (Data2)": "model/resnet_data2.pth", 
        "FocalNet-Large (Data2)": "model/focal_data2.pth",
        "Swin-Large (Data2)": "model/swin_data2.pth"
    }

def check_model_availability(model_name: str) -> bool:
    """Check if model files are available"""
    config_mapping = get_model_config_mapping()
    checkpoint_mapping = get_model_checkpoint_mapping()   
    config_path = config_mapping.get(model_name)
    checkpoint_path = checkpoint_mapping.get(model_name)   
    if not config_path or not checkpoint_path:
        return False    
    return os.path.exists(config_path) and os.path.exists(checkpoint_path)

def create_test_data_loader(dataset, accelerator=None, **kwargs):
    """Create data loader for testing"""
    data_loader = data.DataLoader(
        dataset,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=fixed_generator(),
        **kwargs,
    )
    if accelerator:
        data_loader = accelerator.prepare_data_loader(data_loader)
    return data_loader

def create_temp_annotation(image_path, image_width, image_height, dataset_type="data1"):
    filename = os.path.basename(image_path) 
    if dataset_type == "data1":
        categories = [
            {"id": 1, "name": "cystic_plate"},
            {"id": 2, "name": "calot_triangle"},  
            {"id": 3, "name": "cystic_artery"},
            {"id": 4, "name": "cystic_duct"},
            {"id": 5, "name": "gallbladder"}, 
            {"id": 6, "name": "tool"},
        ]
        description = "Endoscapes2023 dataset"
    else:  # data2
        categories = [
            {"id": 1, "name": "Grasper"},
            {"id": 2, "name": "Bipolar"},
            {"id": 3, "name": "Hook"},
            {"id": 4, "name": "Scissors"},
            {"id": 5, "name": "Clipper"},
            {"id": 6, "name": "Irrigator"},
            {"id": 7, "name": "SpecimenBag"}
        ]
        description = "M2CAI16-tools dataset"
        
    annotation = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2024,
            "contributor": "CeMIT", 
            "date_created": "2024-12-20"
        },
        "licenses": [],
        "images": [{
            "id": 1,
            "file_name": filename,
            "width": image_width,
            "height": image_height
        }],
        "annotations": [],
        "categories": categories
    }
    
    return annotation

def run_single_image_detection_main_test(image_file, model_name, dataset_name, confidence_threshold):
    """Complete single image detection workflow using main_test.py methods"""
    try:
        # Setup environment
        accelerator = Accelerator()
        torch.backends.cudnn.deterministic = True
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            temp_image_path = os.path.join(temp_dir, f"temp_image.jpg")
            image = Image.open(image_file)
            image_width, image_height = image.size
            image.save(temp_image_path)
            
            # Create temporary annotation file
            print("Creating temporary annotation file...")
            annotation_data = create_temp_annotation(temp_image_path, image_width, image_height, dataset_name)
            temp_annotation_path = os.path.join(temp_dir, "annotation.json")
            with open(temp_annotation_path, 'w') as f:
                json.dump(annotation_data, f)
            print(f"Created temporary annotation file: {temp_annotation_path}")
            print(f"Annotation data keys: {annotation_data.keys()}")
            print(f"Number of images: {len(annotation_data['images'])}")
            print(f"Number of categories: {len(annotation_data['categories'])}")
            
            # 1. Load configuration file
            config_file = f"configs/{model_name}_{dataset_name}.py"
            if not os.path.exists(config_file):
                st.error(f"Config file does not exist: {config_file}")
                return None, None, None
                
            config = Config(config_file)
            print(f"Config file loaded: {config_file}")
            
            # 2. Create dataset
            dataset = CocoDetection(
                img_folder=temp_dir,
                ann_file=temp_annotation_path,
                transforms=None  # the eval_transform is integrated in the model
            )
            
            # 3. Create data loader
            data_loader = create_test_data_loader(
                dataset,
                accelerator=accelerator,
                batch_size=1,
                num_workers=0,  # Avoid multiprocessing issues
                collate_fn=collate_fn
            )
            
            # 4. Load model
            model_file = f"model/{model_name}_{dataset_name}.pth"
            if not os.path.exists(model_file):
                st.error(f"Model file does not exist: {model_file}")
                return None, None, None
                
            model = config.model.eval()
            checkpoint = load_checkpoint(model_file)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint = checkpoint["model"]
            load_state_dict(model, checkpoint)
            
            # Prepare model with accelerator
            model = accelerator.prepare_model(model)
            
            # 5. Perform inference
            print("Starting inference...")
            predictions = {}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    print(f"Processing batch {batch_idx + 1}...")
                    images, targets = batch
                    print(f"Image batch shape: {images.shape if hasattr(images, 'shape') else type(images)}")
                    print(f"Number of targets: {len(targets)}")
                    outputs = model(images)
                    print(f"Model outputs keys: {outputs[0].keys() if isinstance(outputs, list) and len(outputs) > 0 else 'No outputs'}")
                    
                    # Process outputs
                    for i, output in enumerate(outputs):
                        print(f"Processing output {i + 1}/{len(outputs)}")
                        print(f"Target keys: {targets[i].keys()}")
                        image_id = targets[i]["image_id"]
                        if hasattr(image_id, 'item'):
                            image_id = image_id.item()
                        else:
                            image_id = int(image_id)
                        print(f"Image ID: {image_id}")
                        
                        # Filter by confidence
                        scores = output["scores"]
                        keep = scores > confidence_threshold
                        
                        filtered_output = {
                            "boxes": output["boxes"][keep].cpu(),
                            "scores": output["scores"][keep].cpu(), 
                            "labels": output["labels"][keep].cpu(),
                            "queries": output.get("queries", torch.zeros(keep.sum()))[keep].cpu()
                        }
                        
                        predictions[image_id] = filtered_output
            
            # 6. Prepare COCO format results for post-processing
            coco_results = []
            ann_id = 1  # Initialize annotation ID counter
            for image_id, prediction in predictions.items():
                if len(prediction["scores"]) == 0:
                    continue
                    
                boxes = prediction["boxes"]
                # Convert to COCO format (x1,y1,x2,y2) -> (x,y,w,h)
                coco_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    # Handle both tensor and scalar cases
                    x1_val = x1.item() if hasattr(x1, 'item') else float(x1)
                    y1_val = y1.item() if hasattr(y1, 'item') else float(y1)
                    w_val = w.item() if hasattr(w, 'item') else float(w)
                    h_val = h.item() if hasattr(h, 'item') else float(h)
                    coco_boxes.append([x1_val, y1_val, w_val, h_val])
                    
                scores = prediction["scores"].tolist()
                labels = prediction["labels"].tolist()
                queries = prediction["queries"].tolist()
                
                for k, box in enumerate(coco_boxes):
                    coco_results.append({
                        "id": ann_id,  # Unique ID for each annotation
                        "image_id": image_id,
                        "category_id": labels[k], 
                        "bbox": box,
                        "score": scores[k],
                        "query": queries[k],
                        "area": box[2] * box[3],  # Add area field (width * height)
                        "iscrowd": 0  # Add iscrowd field
                    })
                    ann_id += 1  # Increment annotation ID
            
            raw_detection_count = len(coco_results)
            print(f"Raw detection count: {raw_detection_count}")
            
            # 7. Create result object for post-processing
            if len(coco_results) > 0:
                # Create a simplified result object
                class SimpleResult:
                    def __init__(self, coco_results, annotation_data):
                        self.dataset = {
                            "annotations": coco_results,
                            "categories": annotation_data["categories"]
                        }
                
                result_obj = SimpleResult(coco_results, annotation_data)
                
                # 8. Post-processing
                print(f"Starting post-processing for {dataset_name}...")
                if dataset_name == "data1":
                    print("Using filter_bbox for data1...")
                    filtered_result = filter_bbox(result_obj)
                else:  # data2
                    print("Using filter_bbox2 for data2...")
                    filtered_result = filter_bbox2(result_obj)
                
                post_processed_count = len(filtered_result.dataset["annotations"])
                print(f"Post-processed detection count: {post_processed_count}")
                
                # Apply confidence threshold filtering
                final_annotations = []
                for ann in filtered_result.dataset["annotations"]:
                    if ann["score"] >= confidence_threshold:
                        final_annotations.append(ann)
                
                final_count = len(final_annotations)
                print(f"Final detection count: {final_count}")
                
                # 9. Visualization
                image_array = np.array(image)
                
                if len(final_annotations) > 0:
                    # Prepare visualization data
                    boxes = []
                    labels = []
                    scores = []
                    
                    for ann in final_annotations:
                        bbox = ann["bbox"]
                        # Convert back to (x1,y1,x2,y2) format
                        x, y, w, h = bbox
                        boxes.append([x, y, x + w, y + h])
                        labels.append(ann["category_id"] - 1)  # Convert to 0-based index
                        scores.append(ann["score"])
                    
                    # Get category names
                    categories = annotation_data["categories"]
                    class_names = [cat["name"] for cat in categories]
                    
                    # Draw bounding boxes
                    visualized_image = plot_bounding_boxes_on_image(
                        image=image_array,
                        boxes=np.array(boxes),
                        labels=np.array(labels),
                        scores=np.array(scores),
                        classes=class_names,
                        show_conf=confidence_threshold,
                        font_scale=1.2,
                        box_thick=2,
                        fill_alpha=0.2,
                        text_box_color=(255, 255, 255),
                        text_alpha=1.0
                    )
                else:
                    visualized_image = image_array
                
                # 10. Save results JSON
                results_json = {
                    "model_name": model_name,
                    "dataset_name": dataset_name, 
                    "confidence_threshold": confidence_threshold,
                    "image_info": {
                        "width": image_width,
                        "height": image_height,
                        "filename": image_file.name
                    },
                    "detection_counts": {
                        "raw_detections": raw_detection_count,
                        "post_processed": post_processed_count,
                        "final_detections": final_count
                    },
                    "detections": final_annotations,
                    "categories": categories
                }
                
                return visualized_image, results_json, final_annotations
                
            else:
                print("No objects detected")
                return np.array(image), {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "confidence_threshold": confidence_threshold,
                    "image_info": {
                        "width": image_width,
                        "height": image_height,
                        "filename": image_file.name
                    },
                    "detection_counts": {
                        "raw_detections": 0,
                        "post_processed": 0,
                        "final_detections": 0
                    },
                    "detections": [],
                    "categories": annotation_data["categories"]
                }, []
                
    except Exception as e:
        print(f"Error occurred during detection: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        st.error(f"Detection failed: {str(e)}")
        return None, None, None

def show_detection_page():
    st.title("🔍 Object Detection - Using LCSOD Tool")
    st.write("Upload an image, select model and data type for object detection")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        # Model selection
        model_options = ["resnet", "focal", "swin"]
        selected_model = st.selectbox(
            "Select Backbone",
            model_options,
            help="Choose the Backbone for LCSOD model"
        )
        
        # Dataset selection
        dataset_options = ["data1", "data2"]
        selected_dataset = st.selectbox(
            "Select Dataset type",
            dataset_options,
            help="Choose the corresponding dataset type"
        )
        
        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only show detection results with confidence above this threshold"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Model Information")
        st.info(f"""
        **Current Configuration:**
        - Model: {selected_model}
        - Dataset: {selected_dataset}
        - Confidence: {confidence}
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Supports JPG, JPEG, PNG formats"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            # st.image(image, caption="Uploaded Image", use_column_width=True)      
            # Display image information
            st.info(f"""
            **Image Information:**
            - Filename: {uploaded_file.name}
            - Dimensions: {image.size[0]} x {image.size[1]}
            - Format: {image.format}
            """)
    
    with col2:
        st.header("🎯 Detection Control")
        
        # Detection button
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("Performing detection..."):
                    # Perform detection
                    visualized_image, results_json, detections = run_single_image_detection_main_test(
                        uploaded_file, selected_model, selected_dataset, confidence
                    )
                    
                    if visualized_image is not None:
                        # Save results to session state
                        st.session_state.detection_results = {
                            'visualized_image': visualized_image,
                            'results_json': results_json,
                            'detections': detections,
                            'original_image': np.array(image)
                        }
                        st.success("✅ Detection completed!")
                        st.rerun()
                    else:
                        st.error("❌ Detection failed, please check configuration and files")
            else:
                st.warning("⚠️ Please upload an image first")
        
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            if 'detection_results' in st.session_state:
                del st.session_state.detection_results
                st.rerun()
    
    # Display detection results
    if 'detection_results' in st.session_state:
        results = st.session_state.detection_results
        
        st.markdown("---")
        st.header("📊 Detection Results")
        
        # Create result tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Visualization Results", "📈 Statistics", "📋 Detailed Detection", "💾 Download Results"])
        
        with tab1:
            st.subheader("Visualization Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Image**")
                st.image(results['original_image'], use_container_width=True)
                
            with col2:
                st.write("**Detection Results**")
                st.image(results['visualized_image'], use_container_width=True)
        
        with tab2:
            st.subheader("Detection Statistics")
            
            if results['results_json']:
                counts = results['results_json']['detection_counts']
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Raw Detections", counts['raw_detections'])
                with col2:
                    st.metric("Post-processed", counts['post_processed']) 
                with col3:
                    st.metric("Final Results", counts['final_detections'])
                
                # Display configuration information
                st.json({
                    "Model Configuration": {
                        "Model": results['results_json']['model_name'],
                        "Dataset": results['results_json']['dataset_name'],
                        "Confidence Threshold": results['results_json']['confidence_threshold']
                    },
                    "Image Information": results['results_json']['image_info']
                })
        
        with tab3:
            st.subheader("Detailed Detection Results")
            
            if results['detections']:
                for i, detection in enumerate(results['detections']):
                    with st.expander(f"Detection {i+1}: {detection.get('category_name', f'Category_{detection['category_id']}')} (Confidence: {detection['score']:.3f})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Bounding Box (x, y, w, h):**")
                            bbox = detection['bbox']
                            st.write(f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
                        with col2:
                            st.write("**Other Information:**")
                            st.write(f"Category ID: {detection['category_id']}")
                            st.write(f"Image ID: {detection['image_id']}")
                            if 'query' in detection:
                                st.write(f"Query: {detection['query']}")
            else:
                st.info("No objects detected")
        
        with tab4:
            st.subheader("Download Results")
            
            if results['results_json']:
                # Prepare JSON data for download
                json_str = json.dumps(results['results_json'], indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Download Detection Results (JSON)",
                    data=json_str,
                    file_name=f"detection_results_{selected_model}_{selected_dataset}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Save visualization result image
                if results['visualized_image'] is not None:
                    # Convert to PIL image for saving
                    pil_image = Image.fromarray(results['visualized_image'])
                    
                    # Convert to byte stream
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    st.download_button(
                        label="🖼️ Download Visualization Result (PNG)",
                        data=img_bytes,
                        file_name=f"detection_visualization_{selected_model}_{selected_dataset}.png",
                        mime="image/png",
                        use_container_width=True
                    )

if __name__ == "__main__":
    show_detection_page()

