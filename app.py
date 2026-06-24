from datasets import load_dataset
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import get_model
from dataset import NUM_CLASSES, CLASS_NAMES
import streamlit as st

st.set_page_config(
    page_title='Clothing Segmentation',
    page_icon='👗',
    layout='wide',
    initial_sidebar_state='expanded'
)

# custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #ccc;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .stAlert { display: none; }
</style>
""", unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = get_model(device=device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_class_color(i):
    cmap = plt.get_cmap('tab20')
    color = cmap(i / (NUM_CLASSES - 1))
    return '#{:02x}{:02x}{:02x}'.format(
        int(color[0]*255), int(color[1]*255), int(color[2]*255))

def segment_image(image, model, alpha=0.5):
    original_size = image.size
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_resized = pred_mask_pil.resize(original_size, Image.NEAREST)
    pred_mask_resized = np.array(pred_mask_resized)
    
    cmap = plt.get_cmap('tab20')
    colored_mask = cmap(pred_mask_resized / (NUM_CLASSES - 1))[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    
    original_np = np.array(image)
    overlay = (original_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    detected = np.unique(pred_mask_resized)
    detected_names = [CLASS_NAMES[i] for i in detected if i < NUM_CLASSES]
    
    return colored_mask, overlay, pred_mask_resized, detected_names

# sidebar
with st.sidebar:
    st.markdown('<p class="main-title">👗 Clothing Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered clothing segmentation</p>', 
                unsafe_allow_html=True)
    
    st.divider()
    
    
    st.markdown('**Class Legend**')
    for i, name in enumerate(CLASS_NAMES):
        color = get_class_color(i)
        st.markdown(f'''
        <div class="legend-item">
            <div style="width:16px;height:16px;border-radius:4px;
                        background:{color};flex-shrink:0"></div>
            <span style="color:#ccc">{name}</span>
        </div>''', unsafe_allow_html=True)

# main area
st.markdown('<h1 class="main-title">Clothing Segmentation</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a photo of a person to instantly segment their clothing items</p>', 
            unsafe_allow_html=True)

model = load_model()

uploaded_file = st.file_uploader(
    'Drop an image here or click to browse',
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed'
)

if uploaded_file is None:
    st.markdown('''
    <div style="border: 2px dashed #444; border-radius: 12px; padding: 60px;
                text-align: center; color: #666; margin-top: 20px;">
        <div style="font-size: 3rem">👆</div>
        <div style="font-size: 1.1rem; margin-top: 12px">
            Upload a person photo to get started
        </div>
        <div style="font-size: 0.85rem; margin-top: 8px; color: #555">
            Supports JPG and PNG
        </div>
    </div>
    ''', unsafe_allow_html=True)

else:
    image = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner('Segmenting clothing...'):
        colored_mask, overlay, pred_mask, detected_names = segment_image(image, model)

    
    # detected classes
    st.markdown('**Detected clothing items:**')
    detected_html = ''
    for name in detected_names:
        idx = CLASS_NAMES.index(name)
        color = get_class_color(idx)
        detected_html += f'''
        <span style="background:{color};color:white;padding:4px 10px;
                     border-radius:20px;font-size:0.8rem;margin:3px;
                     display:inline-block">{name}</span>'''
    st.markdown(detected_html, unsafe_allow_html=True)
    
    st.divider()
    
    # images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-header">Original</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Segmentation Mask</p>', unsafe_allow_html=True)
        st.image(colored_mask, use_container_width=True)

    with col3:
        st.markdown('<p class="section-header">Overlay</p>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
    
    
    st.divider()
    
    # stats
    total_pixels = pred_mask.size
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bg_pct = (pred_mask == 0).sum() / total_pixels * 100
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{100-bg_pct:.1f}%</div>
            <div class="metric-label">Clothing Coverage</div>
        </div>''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{len(detected_names)}</div>
            <div class="metric-label">Classes Detected</div>
        </div>''', unsafe_allow_html=True)
    
    with col3:
        dominant_idx = np.bincount(pred_mask.flatten()).argmax()
        if dominant_idx == 0 and len(detected_names) > 1:
            dominant_idx = np.bincount(
                pred_mask[pred_mask != 0].flatten()).argmax()
        dominant = CLASS_NAMES[dominant_idx] if dominant_idx < NUM_CLASSES else 'Background'
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.2rem">{dominant}</div>
            <div class="metric-label">Dominant Class</div>
        </div>''', unsafe_allow_html=True)