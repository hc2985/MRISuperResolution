"""
Low-Field to High-Field MRI Enhancement Challenge - Evaluation Metric

This metric evaluates predicted MRI slices against ground truth high-field images
using Multi-Scale Structural Similarity (MS-SSIM).

Each row represents a single slice: row_id format is "sample_XXX_slice_YYY"

MS-SSIM evaluates structural quality across 5 resolution scales using
Gaussian-weighted 11x11 windows (sigma=1.5) with standard weights from
Wang, Simoncelli & Bovik (2003).

Score range: 0-1 (higher is better).
"""

import pandas as pd
import pandas.api.types
import numpy as np
import base64
import io


class ParticipantVisibleError(Exception):
    """Error messages shown to participants for debugging their submissions."""
    pass


def base64_to_slice(b64_string: str) -> np.ndarray:
    """
    Decode a base64 string back to a 2D numpy array.

    Args:
        b64_string: Base64 encoded string

    Returns:
        2D numpy array (float32)
    """
    try:
        buffer = io.BytesIO(base64.b64decode(b64_string))
        data = np.load(buffer)

        normalized = data['slice']
        min_val_arr = data['min_val']
        max_val_arr = data['max_val']

        # Handle both scalar and array storage formats
        min_val = float(min_val_arr.item()) if min_val_arr.ndim > 0 else float(min_val_arr)
        max_val = float(max_val_arr.item()) if max_val_arr.ndim > 0 else float(max_val_arr)

        if max_val - min_val > 0:
            original = normalized.astype(np.float32) / 255 * (max_val - min_val) + min_val
        else:
            original = np.zeros_like(normalized, dtype=np.float32)

        return original
    except Exception as e:
        raise ParticipantVisibleError(f"Failed to decode base64 slice: {str(e)}")


def _normalize_01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Create a 2D Gaussian kernel."""
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _ssim_components(img1: np.ndarray, img2: np.ndarray,
                     kernel: np.ndarray) -> tuple:
    """
    Compute SSIM luminance, contrast, and structure components.

    Returns:
        (luminance, contrast_structure) — split as in the MS-SSIM paper:
        luminance = (2*mu1*mu2 + C1) / (mu1^2 + mu2^2 + C1)
        cs        = (2*sigma12 + C2) / (sigma1^2 + sigma2^2 + C2)
    """
    from scipy.signal import fftconvolve

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = fftconvolve(img1, kernel, mode='valid')
    mu2 = fftconvolve(img2, kernel, mode='valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = fftconvolve(img1 * img1, kernel, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(img2 * img2, kernel, mode='valid') - mu2_sq
    sigma12 = fftconvolve(img1 * img2, kernel, mode='valid') - mu1_mu2

    # Clamp variances to avoid numerical issues
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    luminance = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return luminance, cs


def compute_ms_ssim(img1: np.ndarray, img2: np.ndarray,
                    weights: list = None, win_size: int = 11,
                    sigma: float = 1.5) -> float:
    """
    Compute Multi-Scale SSIM between two images.

    Uses 5 scales with standard weights from Wang et al. (2003).
    At each scale, SSIM luminance/contrast/structure components are
    computed with Gaussian-weighted local windows. Images are
    downsampled 2x between scales using average pooling.

    Args:
        img1, img2: 2D numpy arrays (same shape), values in [0, 1]
        weights: Per-scale weights (default: Wang et al. 2003)
        win_size: Gaussian window size (default: 11)
        sigma: Gaussian sigma (default: 1.5)

    Returns:
        MS-SSIM value between 0 and 1
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    n_scales = len(weights)
    kernel = _gaussian_kernel_2d(win_size, sigma).astype(np.float64)

    mcs_list = []
    for scale in range(n_scales):
        # Ensure images are large enough for this scale
        if img1.shape[0] < win_size or img1.shape[1] < win_size:
            # Image too small for more scales; use what we have
            break

        luminance, cs = _ssim_components(img1, img2, kernel)

        if scale == n_scales - 1:
            # Last scale: use full SSIM (luminance * cs)
            mcs_list.append((luminance.mean(), cs.mean()))
        else:
            # Intermediate scales: only keep contrast-structure
            mcs_list.append((None, cs.mean()))

        # Downsample 2x using average pooling
        if scale < n_scales - 1:
            # Trim to even dimensions
            h, w = img1.shape
            h_even, w_even = h - h % 2, w - w % 2
            img1 = img1[:h_even, :w_even].reshape(h_even // 2, 2, w_even // 2, 2).mean(axis=(1, 3))
            img2 = img2[:h_even, :w_even].reshape(h_even // 2, 2, w_even // 2, 2).mean(axis=(1, 3))

    # Compute MS-SSIM product
    n_computed = len(mcs_list)
    if n_computed == 0:
        return 0.0

    # Adjust weights to number of scales actually computed
    used_weights = weights[:n_computed]
    # Renormalize weights
    w_sum = sum(used_weights)
    used_weights = [w / w_sum for w in used_weights]

    ms_ssim = 1.0
    for i, (lum, cs_val) in enumerate(mcs_list):
        # Clamp cs to [0, 1] for stability
        cs_clamped = max(min(cs_val, 1.0), 0.0)
        if i == n_computed - 1 and lum is not None:
            # Last scale: luminance^weight * cs^weight
            lum_clamped = max(min(lum, 1.0), 0.0)
            ms_ssim *= (lum_clamped ** used_weights[i]) * (cs_clamped ** used_weights[i])
        else:
            ms_ssim *= cs_clamped ** used_weights[i]

    return float(ms_ssim)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Evaluate MRI super-resolution predictions using MS-SSIM.

    Each row represents a single slice with format: row_id = "sample_XXX_slice_YYY"

    Final score = mean MS-SSIM across all evaluated slices.

    Higher scores are better. Maximum possible score is 1.0 (perfect prediction).

    Args:
        solution: DataFrame with columns [row_id, ground_truth, Usage]
        submission: DataFrame with columns [row_id, prediction]
        row_id_column_name: Name of the ID column (row_id)

    Returns:
        MS-SSIM score between 0 and 1
    """

    # ========================================
    # VALIDATION: Check submission structure
    # ========================================

    # Check if submission is empty
    if submission is None or len(submission) == 0:
        raise ParticipantVisibleError("Submission is empty. Please provide predictions for all test slices.")

    # Check if required columns exist
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(
            f"Submission must have a '{row_id_column_name}' column. "
            f"Found columns: {list(submission.columns)}"
        )

    if 'prediction' not in submission.columns:
        raise ParticipantVisibleError(
            f"Submission must have a 'prediction' column. "
            f"Found columns: {list(submission.columns)}"
        )

    if 'ground_truth' not in solution.columns:
        raise ParticipantVisibleError("Solution must have a 'ground_truth' column")

    # ========================================
    # VALIDATION: Check for duplicates
    # ========================================

    duplicate_ids = submission[submission[row_id_column_name].duplicated()][row_id_column_name].tolist()
    if duplicate_ids:
        raise ParticipantVisibleError(
            f"Submission contains duplicate row_ids: {duplicate_ids[:5]}{'...' if len(duplicate_ids) > 5 else ''}. "
            f"Each row_id should appear exactly once."
        )

    # ========================================
    # VALIDATION: Check for NaN/empty values
    # ========================================

    # Check for NaN in row_id column
    if submission[row_id_column_name].isna().any():
        raise ParticipantVisibleError(
            f"Submission contains NaN/empty values in '{row_id_column_name}' column. "
            f"All row_ids must be valid."
        )

    # Check for NaN in prediction column
    nan_predictions = submission[submission['prediction'].isna()][row_id_column_name].tolist()
    if nan_predictions:
        raise ParticipantVisibleError(
            f"Submission contains NaN/empty values in 'prediction' column for rows: {nan_predictions[:5]}{'...' if len(nan_predictions) > 5 else ''}. "
            f"All predictions must be valid base64-encoded slices."
        )

    # Check for empty strings in prediction
    empty_predictions = submission[submission['prediction'].astype(str).str.strip() == ''][row_id_column_name].tolist()
    if empty_predictions:
        raise ParticipantVisibleError(
            f"Submission contains empty strings in 'prediction' column for rows: {empty_predictions[:5]}{'...' if len(empty_predictions) > 5 else ''}. "
            f"All predictions must be valid base64-encoded slices."
        )

    # ========================================
    # MERGE: Align submission with solution
    # ========================================

    merged = solution.merge(submission, on=row_id_column_name, how='left')

    # Check for missing predictions after merge
    missing_mask = merged['prediction'].isna()
    if missing_mask.any():
        missing_ids = merged[missing_mask][row_id_column_name].tolist()
        raise ParticipantVisibleError(
            f"Missing predictions for {len(missing_ids)} rows: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}. "
            f"Your submission must include predictions for all test slices."
        )

    # ========================================
    # COMPUTE MS-SSIM
    # ========================================

    all_ms_ssim = []

    for idx, row in merged.iterrows():
        row_id = row[row_id_column_name]

        # Decode ground truth
        try:
            gt_slice = base64_to_slice(row['ground_truth'])
        except ParticipantVisibleError:
            raise
        except Exception as e:
            raise ParticipantVisibleError(f"Error decoding solution for {row_id}: {str(e)}")

        # Decode prediction
        try:
            pred_slice = base64_to_slice(row['prediction'])
        except ParticipantVisibleError as e:
            raise ParticipantVisibleError(f"Error decoding prediction for {row_id}: {str(e)}")
        except Exception as e:
            raise ParticipantVisibleError(
                f"Error decoding prediction for {row_id}: {str(e)}. "
                f"Make sure you're using the provided slice_to_base64() function."
            )

        # Check shape match
        expected_shape = (179, 221)
        if pred_slice.shape != expected_shape:
            raise ParticipantVisibleError(
                f"Shape mismatch for {row_id}: expected {expected_shape}, got {pred_slice.shape}. "
                f"Your predicted slices must be exactly (179, 221) pixels."
            )

        if gt_slice.shape != pred_slice.shape:
            raise ParticipantVisibleError(
                f"Shape mismatch for {row_id}: ground truth is {gt_slice.shape}, prediction is {pred_slice.shape}."
            )

        # Check for NaN/Inf in prediction
        if np.isnan(pred_slice).any() or np.isinf(pred_slice).any():
            raise ParticipantVisibleError(
                f"Prediction for {row_id} contains NaN or Inf values. "
                f"All pixel values must be finite numbers."
            )

        # Normalize both to [0, 1]
        gt_norm = _normalize_01(gt_slice).astype(np.float64)
        pred_norm = _normalize_01(pred_slice).astype(np.float64)

        # Compute MS-SSIM
        ms_ssim = compute_ms_ssim(gt_norm, pred_norm)
        all_ms_ssim.append(ms_ssim)

    # ========================================
    # COMPUTE FINAL SCORE
    # ========================================

    final_score = float(np.mean(all_ms_ssim))

    # Ensure score is finite
    if not np.isfinite(final_score):
        raise ParticipantVisibleError("Computed score is not finite. Please check your predictions.")

    return final_score