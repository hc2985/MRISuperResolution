from extract_slices import create_submission_df, volume_to_submission_rows
from model import model
import pandas as pd


low_field_input = "/train/low_field/sample_001_lowfield.nii"
predicted_volume = model(low_field_input)  # Shape: (179, 221, 200)
rows = volume_to_submission_rows(predicted_volume, 'sample_019')

