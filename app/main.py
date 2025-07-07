import streamlit as st
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import pages
from pages import detection

# Configure page
st.set_page_config(
    page_title="LCSOD Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's default navigation
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.css-1d391kg {padding-top: 1rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.title("🎯 LCSOD Tool")
    st.sidebar.markdown("---")
    
    # Navigation options
    pages = {
        "🏠 Home": "home",
        "🎯 Object Detection": "detection"
    }
    
    # Create navigation with session state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "🏠 Home"
    
    selected_page = st.sidebar.radio(
        "Navigate to:",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.selected_page)
    )
    
    # Update session state when selection changes
    st.session_state.selected_page = selected_page
    
    # Device info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💻 Device Info")
    
    import torch
    if torch.cuda.is_available():
        st.sidebar.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        st.sidebar.write(f"Memory: {memory_gb}GB")
    else:
        st.sidebar.warning("⚠️ Using CPU")
    
    # Main content
    if selected_page == "🏠 Home":
        show_home()
    elif selected_page == "🎯 Object Detection":
        detection.show_detection_page()

def show_home():
    """Show home page"""
    st.title("🎯 LCSOD Tool")
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ## Welcome to LCSOD Tool
    
    This application provides object detection capabilities using state-of-the-art models.
    Upload an image and select a model to perform object detection analysis.
    """)
    
    # Two-column layout for models and datasets
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Available Models
        - **ResNet-50**: Fast and efficient backbone
        - **FocalNet-Large**: High-accuracy backbone  
        - **Swin-Large**: Balanced detection backbone
        """)
    
    with col2:
        st.markdown("""
        ### Supported Datasets
        - **Data1**: Endoscapes2023-BBox201
        - **Data2**: M2cai16-tool-locations
        """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        ### ✨ Features
        - Upload custom images
        - Multiple model selection
        - Real-time detection results
        - Interactive visualization
        - Confidence threshold adjustment
        """)
    
    with col4:
        st.markdown("""
        ### 🚀 How to Use
        1. Navigate to **Object Detection** page
        2. Upload an image file
        3. Select a model and dataset
        4. Adjust confidence threshold
        5. Click **Run Detection**
        """)
    
    # Quick start
    st.markdown("---")
    st.markdown("### 🎯 Quick Start")
    
    if st.button("🚀 Go to Object Detection", type="primary"):
        st.session_state.selected_page = "🎯 Object Detection"
        st.rerun()

if __name__ == "__main__":
    main() 