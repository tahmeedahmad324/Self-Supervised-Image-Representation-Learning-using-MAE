import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="MAE Reconstruction", layout="wide")

IMG_SIZE = 224
PATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Architecture
class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=depth,
            norm=norm_layer(embed_dim)
        )
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=decoder_depth,
            norm=norm_layer(decoder_embed_dim)
        )
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)
        
    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, imgs):
        # Patch embedding
        x = self.patch_embed(imgs)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Target for reconstruction
        target = patchify(imgs, self.patch_size)
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x + self.decoder_pos_embed
        
        x = self.decoder(x)
        x = self.decoder_pred(x)
        
        return x, mask, target

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
    """Load the trained model"""
    try:
        # Initialize model architecture
        model = MAE(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mask_ratio=0.75
        ).to(device)
        
        # Load state dict
        state_dict = torch.load('mae_best_model.pth', map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'mae_best_model.pth' not found!")
        st.info("Make sure you've saved the best model in Kaggle using: torch.save(model.state_dict(), 'mae_best_model.pth')")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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