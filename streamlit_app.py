import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="MAE Reconstruction", layout="wide")

IMG_SIZE = 224
PATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility functions (lightweight, needed for preprocessing)
def patchify(images, patch_size):
    B, C, H, W = images.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    patches = images.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    patches = patches.reshape(B, num_patches_h * num_patches_w, patch_size * patch_size * C)
    return patches

def unpatchify(patches, patch_size, img_size):
    B, num_patches, _ = patches.shape
    C = 3
    num_patches_side = img_size // patch_size
    patches = patches.reshape(B, num_patches_side, num_patches_side, patch_size, patch_size, C)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    images = patches.reshape(B, C, img_size, img_size)
    return images

@st.cache_resource
def load_model():
    """Load the complete trained model"""
    try:
        # Load complete model (saved with torch.save(model, ...))
        model = torch.load('mae_complete_model.pth', map_location=device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'mae_complete_model.pth' not found!")
        st.info("Make sure you've saved the complete model in Kaggle using: torch.save(model, 'mae_complete_model.pth')")
        st.stop()

model = load_model()

# UI
st.title("🎨 Masked Autoencoder (MAE)")
st.markdown("### Self-Supervised Image Reconstruction")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)
    
    mask_ratio = st.slider(
        "Masking Ratio",
        min_value=0.25,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Percentage of patches to mask"
    )
    
    reconstruct_btn = st.button("🔄 Reconstruct", type="primary", use_container_width=True)

with col2:
    st.subheader("📥 Output")
    
    if uploaded_file and reconstruct_btn:
        with st.spinner("Reconstructing image..."):
            # Preprocess
            transform_test = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform_test(image).unsqueeze(0).to(device)
            
            # Update mask ratio
            model.mask_ratio = mask_ratio
            
            # Generate reconstruction
            with torch.no_grad():
                pred, mask, target = model(img_tensor)
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            # Reconstructed image
            pred_img = unpatchify(pred, PATCH_SIZE, IMG_SIZE)
            pred_img = pred_img * std + mean
            pred_img = torch.clamp(pred_img, 0, 1)
            
            # Masked image
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, PATCH_SIZE**2 * 3)
            masked_patches = target * (1 - mask_expanded)
            masked_img = unpatchify(masked_patches, PATCH_SIZE, IMG_SIZE)
            masked_img = masked_img * std + mean
            masked_img = torch.clamp(masked_img, 0, 1)
            
            # Convert to numpy
            result = pred_img[0].cpu().permute(1, 2, 0).numpy()
            masked = masked_img[0].cpu().permute(1, 2, 0).numpy()
            
            # Display
            tab1, tab2 = st.tabs(["Masked Input", "Reconstruction"])
            
            with tab1:
                st.image(masked, caption=f"Masked ({int(mask_ratio*100)}%)", use_container_width=True)
            
            with tab2:
                st.image(result, caption="Reconstructed Image", use_container_width=True)
    else:
        st.info("👆 Upload an image and click Reconstruct")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Generative AI Assignment 2 | AI4009 | Spring 2026</p>
</div>
""", unsafe_allow_html=True)