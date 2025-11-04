import numpy as np
import os
from typing import Tuple, Optional
counter = 0
brvc = '/mnt/d/repos/brvc/'
rvc = '/mnt/d/repos/rvc-webgui-fork/'
def compare_arrays(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[str, Optional[float]]:
    global counter
    counter += 1
    print(f"Comparison #{counter}:")
    print(f"--- Comparing Array 1 (Length: {len(arr1)}) and Array 2 (Length: {len(arr2)}) ---")

    # 1. Compare Lengths
    if len(arr1) != len(arr2):
        length_diff = abs(len(arr1) - len(arr2))
        message = (
            f"Arrays have different lengths. Length difference: {length_diff}. "
            "Cannot calculate item-by-item difference sum."
        )
        return message, None
    else:
        # 2. Calculate the difference between each item and sum it up
        # np.abs(arr1 - arr2) calculates the absolute difference for each corresponding element.
        # np.sum() then adds all these absolute differences together.
        difference_sum = np.sum(np.abs(arr1 - arr2))

        message = (
            f"Arrays have the same length ({len(arr1)}). "
            "Item-by-item difference summation complete."
        )
        return message, difference_sum
    

# --- How to use this with your actual file paths ---

def load_and_compare(filepath1: str, filepath2: str):
    """Loads arrays from file paths and compares them."""
    try:
        # Check if files exist (optional but good practice)
        if not os.path.exists(filepath1):
            print(f"Error: File not found at {filepath1}")
            return
        if not os.path.exists(filepath2):
            print(f"Error: File not found at {filepath2}")
            return

        # Load the arrays from the .npy files
        actual_arr1 = np.load(filepath1)
        actual_arr2 = np.load(filepath2)
        
        print(f"=============\nLoaded arrays from:\n  {filepath1}\n  {filepath2}")

        # Perform comparison
        message, difference_sum = compare_arrays(actual_arr1, actual_arr2)
        print(message)
        if difference_sum is not None:
            print(f"Actual Sum of Absolute Differences: {difference_sum:.4f}")
        print("--------------------------------------------------\n")
    except Exception as e:
        print(f"An error occurred during file loading or comparison: {e}")

# Example of how you would call it with your original paths:
# Note: Uncomment the line below and ensure the files exist if running locally.
load_and_compare(f'{brvc}debug_loaded_audio.npy', f'{rvc}debug_loaded_audio.npy')
load_and_compare(f'{brvc}debug_filtered_audio.npy', f'{rvc}debug_filtered_audio.npy')
load_and_compare(f'{brvc}debug_f0bak.npy', f'{rvc}debug_f0bak.npy')
load_and_compare(f'{brvc}debug_f0_coarse.npy', f'{rvc}debug_f0_coarse.npy')
load_and_compare(f'{brvc}debug_feats.npy', f'{rvc}debug_feats.npy')
load_and_compare(f'{brvc}debug_feats_interp.npy', f'{rvc}debug_feats_interp.npy')
load_and_compare(f'{brvc}debug_audio_opt_array.npy', f'{rvc}debug_audio_opt_array.npy')
load_and_compare(f'{brvc}debug_final_output_audio.npy', f'{rvc}debug_final_output_audio.npy')
load_and_compare(f'{brvc}debug_converted_audio.npy', f'{rvc}debug_converted_audio.npy')

# Load wavs for audio comparison
import soundfile as sf
def load_wav_and_compare(filepath1: str, filepath2: str):
    """Loads wav files from file paths and compares them."""
    try:
        # Check if files exist (optional but good practice)
        if not os.path.exists(filepath1):
            print(f"Error: File not found at {filepath1}")
            return
        if not os.path.exists(filepath2):
            print(f"Error: File not found at {filepath2}")
            return

        # Load the wav files
        actual_wav1, sr1 = sf.read(filepath1)
        actual_wav2, sr2 = sf.read(filepath2)

        if sr1 != sr2:
            print(f"Sample rates differ: {sr1} vs {sr2}. Cannot compare audio.")
            return
        print(f"=============\nLoaded wavs from:\n  {filepath1} (SR: {sr1})\n  {filepath2} (SR: {sr2})")
        # Perform comparison
        message, difference_sum = compare_arrays(actual_wav1, actual_wav2)
        print(message)
        if difference_sum is not None:
            print(f"Actual Sum of Absolute Differences: {difference_sum:.4f}")
        print("--------------------------------------------------\n")

    except Exception as e:
        print(f"An error occurred during file loading or comparison: {e}")

load_wav_and_compare(f'{brvc}ttt_out.wav', f'{rvc}ttt_out.wav')
load_wav_and_compare(f'{brvc}debug_filtered_audio.wav', f'{rvc}debug_filtered_audio.wav')