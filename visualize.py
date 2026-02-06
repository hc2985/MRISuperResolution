from extract_slices import load_nifti, base64_to_slice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load submission CSV
submission_path = "submission.csv"
df = pd.read_csv(submission_path)
print(f"Loaded {len(df)} rows from {submission_path}")

# Test samples
test_dir = "test/low_field"
samples = range(19, 24)

# Pick slices to visualize (evenly spaced through the volume)
slice_indices = [30, 60, 100, 140, 170]

for sample_idx in samples:
    sample_id = f"sample_{sample_idx:03d}"
    lf_path = os.path.join(test_dir, f"{sample_id}_lowfield.nii")
    lf_vol = load_nifti(lf_path)  # (112, 138, 40)

    fig, axes = plt.subplots(2, len(slice_indices), figsize=(4 * len(slice_indices), 8))
    fig.suptitle(f"{sample_id} — Low-field (top) vs Predicted (bottom)", fontsize=14)

    for col, sl in enumerate(slice_indices):
        # Low-field: map slice index from 200 range to 40 range
        lf_sl = int(sl * (lf_vol.shape[2] / 200))
        lf_slice = lf_vol[:, :, lf_sl]
        axes[0, col].imshow(lf_slice, cmap='gray')
        axes[0, col].set_title(f"LF slice {lf_sl}")
        axes[0, col].axis('off')

        # Predicted: decode from CSV
        row_id = f"{sample_id}_slice_{sl:03d}"
        row = df[df['row_id'] == row_id]
        if len(row) > 0:
            pred_slice = base64_to_slice(row.iloc[0]['prediction'])
            axes[1, col].imshow(pred_slice, cmap='gray')
            axes[1, col].set_title(f"Pred slice {sl}")
        axes[1, col].axis('off')

    plt.tight_layout()
    out_path = f"compare_{sample_id}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved {out_path}")

print("Done")
