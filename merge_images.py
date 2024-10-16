import numpy as np
import imageio.v2 as imageio
import cv2
import os
import re

# Set directory paths
in_dir = "/Users/laurahuggon/Library/CloudStorage/OneDrive-King'sCollegeLondon/phd/lab/imaging/echo/imaging_data_y1/neuron_quantification/neun/ilastik/dapi_tritc"
out_dir = "/Users/laurahuggon/Library/CloudStorage/OneDrive-King'sCollegeLondon/phd/lab/imaging/echo/imaging_data_y1/neuron_quantification/neun/ilastik/dapi_tritc_merge"

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Regular expression to match the image filename pattern
# Match the prefix "NEUN_BTUB_MAP2_DIFF" followed by any number of characters (.*) and an underscore (_)
# This is followed by any number of characters (.*) and an underscore (_)
# This is followed by exactly four digits (\d{4}) and an underscore (_)
# This is followed by either dapi or tritc and ends in .JPG (the backslash escapes the dot to match a literal period)
pattern = r'NEUN_BTUB_MAP2_DIFF(.*)_(.*)_(\d{4})_(dapi|tritc)\.JPG'

# Dictionary to store the matched pairs of images - the keys will be unique identifiers from the filenames and the values will be another dictionary containing paths to the corresponding images
image_pairs = {}

# Iterate through all files in the directory
for filename in os.listdir(in_dir): # List all files and directories in the `in_dir` directory and iterate over each item
    # If `filename` matches the `pattern`, `match` will be a `Match` object containing the captured groups; otherwise it will be `None`
    match = re.match(pattern, filename) # Attempts to match the filename against the previously defined `pattern`
    if match: # If `match` is not `None`, the filename adheres to the expected format and should be processed further
        # Extract the part that defines the unique image
        image_key = f'{match.group(1)}_{match.group(2)}_{match.group(3)}' # Format the `image_key` from the first three captured groups from the regex
        channel = match.group(4) # Format the `channel` from the fourth captured groups from the regex

        # Add the file to the corresponding channel (dapi or map2)
        if image_key not in image_pairs: # Checks if the `image_key` is already a key in the `image_pairs` dictionary
            image_pairs[image_key] = {} # If not, add the `image_key` as a key in the `image_pairs` dictionary and assign the value as an empty dictionary
        # Add the `channel` as a key in the empty dictionary and assign the value as the full path
        image_pairs[image_key][channel] = os.path.join(in_dir, filename) # Construct the full path to the image file by join the input directory path with the filename
# After this loop, `image_pairs` will contain entries where each key corresponds to a unique image set, and each value is a dictionary with keys 'dapi' and 'tritc' pointing to their respective image file paths

# Iterate over each key-value pair in `image_pairs`, where the key is `image_key` (containing the unique identifier) and the value is `channels` (containing a dictionary)
for image_key, channels in image_pairs.items():
    # Check if both the 'dapi' and 'tritc' keys are present in the `channels` dictionary for the current `image_key`
    if 'dapi' in channels and 'tritc' in channels:
        # Load both DAPI and MAP2 images
        dapi_image = imageio.imread(channels['dapi']) # channels['dapi'] retrieves the file path to the DAPI image
        marker_image = imageio.imread(channels['tritc']) # channels['tritc'] retrieves the file path to the marker image

        # Ensure DAPI is a single-channel image (grayscale)
        if dapi_image.ndim == 3 and dapi_image.shape[2] == 3:  # RGB image
            dapi_image = cv2.cvtColor(dapi_image, cv2.COLOR_RGB2GRAY)
        elif dapi_image.ndim == 3 and dapi_image.shape[2] == 4:  # RGBA image
            dapi_image = cv2.cvtColor(dapi_image, cv2.COLOR_RGBA2GRAY)

        # Ensure marker is a single-channel image (grayscale)
        if marker_image.ndim == 3 and marker_image.shape[2] == 3:  # RGB image
            marker_image = cv2.cvtColor(marker_image, cv2.COLOR_RGB2GRAY)
        elif marker_image.ndim == 3 and marker_image.shape[2] == 4:  # RGBA image
            marker_image = cv2.cvtColor(marker_image, cv2.COLOR_RGBA2GRAY)

        # Ensure both images are the same shape
        if dapi_image.shape != marker_image.shape:
            raise ValueError(f"DAPI and marker images for {image_key} must have the same dimensions.")

        # Initialize a new array for the merged image with only 2 channels
        # Create a new NumPy array filled with zeros with the same height (dapi_image.shape[0]) and width (dapi_image.shape[1]) as the DAPI image with three colour channels
        merged_image = np.zeros((dapi_image.shape[0], dapi_image.shape[1], 3), dtype=np.uint8)

        # Assign the DAPI (blue) and marker (red) channels
        # ... represents all dimensions except the last one
        merged_image[..., 0] = marker_image  # Red channel
        merged_image[..., 1] = 0  # Green channel (set to 0, not used)
        merged_image[..., 2] = dapi_image  # Blue channel

        # Save the multi-channel image
        output_path = os.path.join(out_dir, f'merged_image_{image_key}.tif')
        imageio.imwrite(output_path, merged_image)

        print(f"Merged image {output_path} saved successfully.")
    else:
        print(f"Skipping image pair {image_key}, missing DAPI or MAP2 (tritc) image.")