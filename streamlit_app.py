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

# Model Architecture (matching training code structure)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Patchify and embed
        x = patchify(x, self.patch_size)  # B, num_patches, patch_dim
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_patches, encoder_dim=768, decoder_dim=384, depth=8, num_heads=16, patch_size=16):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size**2 * 3)
        
    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pred(x)
        return x

class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 decoder_dim=384, decoder_depth=8, decoder_num_heads=16, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        self.encoder = Encoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.decoder = Decoder(self.num_patches, embed_dim, decoder_dim, decoder_depth, decoder_num_heads, patch_size)
        
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
        target = patchify(imgs, self.patch_size)
        x = self.encoder(imgs)
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        pred = self.decoder(x, ids_restore)
        return pred, mask, target

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
        # Initialize model architecture (matching training configuration)
        model = MAE(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_dim=384,  # Corrected decoder dimension
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