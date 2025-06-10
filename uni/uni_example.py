import os
import torch
import numpy as np
from torchvision import transforms
import timm
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Configuration ---

# Path to the local directory for model weights
local_dir_models = "../assets/ckpts/uni2-h/"
os.makedirs(local_dir_models, exist_ok=True)


# Download UNI model weights if needed
# from huggingface_hub import hf_hub_download
#login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
# hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
model_weights_path = os.path.join(local_dir_models, "pytorch_model.bin")

# Define the path to input image
image_path = "../hpc-data/lung/one/raw/CytAssist_11mm_FFPE_Human_Lung_Cancer_image.tif" 

# Model parameters
timm_kwargs = {
    'model_name': 'vit_giant_patch14_224',
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}

# Image processing transform (for model input)
# The model expects 224x224 input after this transform
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Patching parameters for the large image
PATCH_SIZE = 224 # Size of patches to extract from the large image for processing
STRIDE = 224     # Stride for non-overlapping patches. Use smaller for overlap.
NUM_CLUSTERS = 5 # Number of regions/clusters to find

# --- Set Device ---
# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# --- Load Model ---
try:
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    # Load state dict first, then move to device
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"), strict=True)
    model.to(device) # Move model to the selected device
    model.eval()
    print("UNI model loaded successfully and moved to device.")
except FileNotFoundError:
    print(f"Error: Model weights not found at {model_weights_path}.")
    print("Please ensure you have downloaded the 'pytorch_model.bin' file and updated 'local_dir'.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- Load and Process Image Patches ---
try:
    original_image = Image.open(image_path).convert("RGB")
    print(f"Original image loaded with size: {original_image.size}")
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    exit()
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

image_width, image_height = original_image.size

patch_features = []
patch_coords = []

print("Extracting features from image patches...")
with torch.inference_mode():
    for y in range(0, image_height, STRIDE):
        for x in range(0, image_width, STRIDE):
            # Define the bounding box for the patch
            box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)

            # Crop the patch
            patch = original_image.crop(box)

            # Check if the patch is smaller than PATCH_SIZE x PATCH_SIZE at the edges
            # If so, pad it
            if patch.size[0] < PATCH_SIZE or patch.size[1] < PATCH_SIZE:
                 # Create a new blank image (e.g., white or black) and paste the patch
                 padded_patch = Image.new('RGB', (PATCH_SIZE, PATCH_SIZE), (255, 255, 255)) # Pad with white
                 padded_patch.paste(patch, (0, 0))
                 patch = padded_patch

            # Apply the model transform to the patch
            # The transform includes resizing to 224x224, ToTensor, and Normalize
            input_tensor = transform(patch).unsqueeze(0) # Add batch dimension
            input_tensor = input_tensor.to(device) # Move input tensor to the selected device

            # Extract features using the UNI model
            features = model(input_tensor) # Shape [1, 1536]

            # Store features and original coordinates
            # Move features back to CPU before converting to numpy for scikit-learn
            patch_features.append(features.squeeze(0).cpu().numpy())
            patch_coords.append((x, y))

print(f"Extracted features for {len(patch_features)} patches.")

# Convert list of features to a numpy array
if not patch_features:
    print("No patches were processed. Please check image path and patch/stride sizes.")
    exit()
all_patch_features = np.array(patch_features)

# --- Perform Clustering ---
print(f"Performing K-Means clustering with {NUM_CLUSTERS} clusters...")
try:
    # KMeans from scikit-learn typically runs on CPU
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10) # n_init='auto' in newer sklearn
    cluster_labels = kmeans.fit_predict(all_patch_features)
    print("Clustering complete.")
except Exception as e:
    print(f"Error during clustering: {e}")
    exit()

# --- Create Overlay ---
print("Creating overlay image...")

# Generate distinct colors for each cluster
colors = plt.cm.get_cmap('tab10', NUM_CLUSTERS) # Using a matplotlib colormap
cluster_colors = [tuple(int(c * 255) for c in colors(i)[:3]) for i in range(NUM_CLUSTERS)]

# Create a transparent overlay layer
overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Draw colored rectangles on the overlay based on cluster labels
for i, (x, y) in enumerate(patch_coords):
    cluster_id = cluster_labels[i]
    color = cluster_colors[cluster_id] + (128,) # Add alpha channel for semi-transparency (128 out of 255)

    # Draw a rectangle for the patch area
    draw.rectangle([x, y, x + PATCH_SIZE, y + PATCH_SIZE], fill=color)

print("Overlay created.")

# --- Blend Images ---
# Blend the original image and the overlay
# We need to convert the original image to RGBA to blend properly
original_image_rgba = original_image.convert('RGBA')
blended_image = Image.alpha_composite(original_image_rgba, overlay)

print("Images blended.")

# --- Save or Display Result ---
output_path = "outputs"
os.makedirs(output_path, exist_ok=True)

try:
    blended_image.save(os.path.join(output_path, "cluster_overlay.png"), format="PNG")
    print(f"Result saved to {output_path}")
except Exception as e:
    print(f"Error saving image: {e}")

# Display the image
blended_image.show()