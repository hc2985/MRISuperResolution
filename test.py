from extract_slices import load_nifti, create_submission_df
from rbdunet import Generator, Discriminator, generator_loss, discriminator_loss
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================
# Load Training Data
# =====================

train_dir = "train"
low_field_dir = os.path.join(train_dir, "low_field")
high_field_dir = os.path.join(train_dir, "high_field")

def load_volume_as_tensor(path):
    """Load NIfTI, normalize to [-1, 1], convert to (1, 1, D, H, W) tensor."""
    vol = load_nifti(path)                  # (H, W, D) numpy
    vol = vol / vol.max()                   # normalize to [0, 1]
    vol = vol * 2 - 1                       # scale to [-1, 1] (match tanh output)
    vol = np.transpose(vol, (2, 0, 1))      # (D, H, W)
    tensor = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor

print("Loading training data...")
train_pairs = []
for i in range(1, 19):  # samples 001-018
    lf_path = os.path.join(low_field_dir, f"sample_{i:03d}_lowfield.nii")
    hf_path = os.path.join(high_field_dir, f"sample_{i:03d}_highfield.nii")
    lf = load_volume_as_tensor(lf_path)  # (1, 1, 40, 112, 138)
    hf = load_volume_as_tensor(hf_path)  # (1, 1, 200, 179, 221)
    train_pairs.append((lf, hf))
    print(f"  Loaded sample {i:03d}")
print(f"Loaded {len(train_pairs)} training pairs")

# =====================
# Initialize Models
# ===================== 
generator = Generator().to(device)
discriminator = Discriminator().to(device)

lr = 0.0002
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00003, betas=(0.5, 0.999))

# =====================
# Training Loop
# =====================

num_epochs = 100

for epoch in range(num_epochs):
    g_losses = []
    d_losses = []

    for i, (low_field, high_field) in enumerate(train_pairs):
        low_field = low_field.to(device)
        high_field = high_field.to(device)

        # Generate fake high-field
        fake = generator(low_field)

        # --- Train Discriminator ---
        d_optimizer.zero_grad()
        d_loss = discriminator_loss(discriminator, fake, high_field)
        d_loss.backward()
        d_optimizer.step()

        # --- Train Generator ---
        g_optimizer.zero_grad()
        fake = generator(low_field)  # regenerate after discriminator update
        g_loss = generator_loss(discriminator, fake, high_field)
        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    avg_g = np.mean(g_losses)
    avg_d = np.mean(d_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}] G_loss: {avg_g:.4f} D_loss: {avg_d:.4f}")

print("Training complete")

# =====================
# Inference on Test Set
# =====================

print("Running inference on test set...")
test_dir = "test/low_field"
generator.eval()

predictions = {}
with torch.no_grad():
    for i in range(19, 24):  # samples 019-023
        lf_path = os.path.join(test_dir, f"sample_{i:03d}_lowfield.nii")
        lf = load_volume_as_tensor(lf_path).to(device)

        fake = generator(lf)  # (1, 1, 200, 179, 221)

        # Convert back to numpy (179, 221, 200) for submission
        vol = fake.squeeze().cpu().numpy()      # (200, 179, 221)
        vol = (vol + 1) / 2                      # [-1,1] → [0,1]
        vol = np.transpose(vol, (1, 2, 0))       # (179, 221, 200)

        sample_id = f"sample_{i:03d}"
        predictions[sample_id] = vol
        print(f"  Predicted {sample_id}")

# =====================
# Create Submission
# =====================

submission_df = create_submission_df(predictions)
submission_df.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission_df)} rows")
