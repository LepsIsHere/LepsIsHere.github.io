import numpy as np
from skimage import io
from skimage.transform import rescale
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import os
import time

def align(channel1, channel2, search_window=(-15, 15)):
    """
    Exhaustively searches for the best alignment of channel2 to channel1.
    
    Args:
        channel1 (np.array): The reference channel (e.g., Blue).
        channel2 (np.array): The channel to align (e.g., Green or Red).
        search_window (tuple): A range of (x, y) pixel displacements to search.

    Returns:
        tuple: The best (y, x) displacement vector.
    """
    min_ssd = float('inf')
    best_displacement = (0, 0)
    
    # Crop 20% of the border to avoid edge effects in metric calculation
    crop_h = int(channel1.shape[0] * 0.2)
    crop_w = int(channel1.shape[1] * 0.2)
    cropped_c1 = channel1[crop_h:-crop_h, crop_w:-crop_w]

    for dy in range(search_window[0], search_window[1] + 1):
        for dx in range(search_window[0], search_window[1] + 1):
            # Roll channel2 to apply the current displacement
            shifted_c2 = np.roll(channel2, (dy, dx), axis=(0, 1))
            
            # Crop the shifted channel2 to match the size and location of cropped_c1
            cropped_c2 = shifted_c2[crop_h:-crop_h, crop_w:-crop_w]
            
            # Calculate Sum of Squared Differences (SSD)
            ssd = np.sum((cropped_c1 - cropped_c2) ** 2)
            
            if ssd < min_ssd:
                min_ssd = ssd
                best_displacement = (dy, dx)
                
    return best_displacement

def pyramid_align(channel1, channel2, max_levels=4):
    """
    Aligns two channels using a coarse-to-fine image pyramid.

    Args:
        channel1 (np.array): The reference channel.
        channel2 (np.array): The channel to align.
        max_levels (int): The number of pyramid levels to use.

    Returns:
        tuple: The final (y, x) displacement vector.
    """
    if max_levels == 0 or min(channel1.shape) < 200: # Base case for recursion
        return align(channel1, channel2, search_window=(-15, 15))

    # Downscale images for the next pyramid level
    scaled_c1 = rescale(channel1, 0.5, anti_aliasing=True)
    scaled_c2 = rescale(channel2, 0.5, anti_aliasing=True)
    
    # Recursively find displacement at the coarser scale
    coarse_displacement = pyramid_align(scaled_c1, scaled_c2, max_levels - 1)
    
    # Scale displacement up for the current level
    scaled_displacement = (coarse_displacement[0] * 2, coarse_displacement[1] * 2)
    
    # Refine the alignment in a smaller search window around the scaled estimate
    # We roll channel2 *before* the fine search
    rolled_c2 = np.roll(channel2, scaled_displacement, axis=(0, 1))
    
    # Search in a small window (e.g., -2 to 2) to refine the coarse estimate
    fine_displacement = align(channel1, rolled_c2, search_window=(-2, 2))
    
    # The final displacement is the sum of the coarse (scaled) and fine adjustments
    final_displacement = (scaled_displacement[0] + fine_displacement[0], 
                          scaled_displacement[1] + fine_displacement[1])
    
    return final_displacement

def auto_contrast(image):
    """Performs automatic contrast stretching."""
    # Find the 2nd and 98th percentiles to avoid outliers
    low_p, high_p = np.percentile(image, (2, 98))
    return np.clip((image - low_p) / (high_p - low_p), 0, 1)

def auto_white_balance(image):
    """Performs automatic white balance using the gray world assumption."""
    # Calculate the average value for each channel
    avg_r, avg_g, avg_b = np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])
    
    # The average of the three channel averages is our target "gray" and calculate scaling factors
    avg_gray = (avg_r + avg_g + avg_b) / 3
    scale_r, scale_g, scale_b = avg_gray / avg_r, avg_gray / avg_g, avg_gray / avg_b
    
    # Apply the scaling
    balanced_image = image.copy()
    balanced_image[:,:,0] *= scale_r
    balanced_image[:,:,1] *= scale_g
    balanced_image[:,:,2] *= scale_b
    
    return np.clip(balanced_image, 0, 1)

def process_image(filepath, output_dir):
    """
    Loads an image, aligns the channels, applies enhancements, and saves it.
    """
    print(f"Processing {os.path.basename(filepath)}...")
    start_time = time.time()
    
    # Load and convert image to float
    img = img_as_float(io.imread(filepath))
    
    # Split the image into B, G, R channels and then align themm
    h = img.shape[0] // 3
    b_ch = img[0:h, :]
    g_ch = img[h:2*h, :]
    r_ch = img[2*h:3*h, :]
    
    g_displacement = pyramid_align(b_ch, g_ch)
    r_displacement = pyramid_align(b_ch, r_ch)
    
    print(f"  - Green channel displacement (y, x): {g_displacement}")
    print(f"  - Red channel displacement (y, x): {r_displacement}")
    
    g_aligned = np.roll(g_ch, g_displacement, axis=(0, 1))
    r_aligned = np.roll(r_ch, r_displacement, axis=(0, 1))
    

    aligned_image = np.dstack([r_aligned, g_aligned, b_ch])
    
    # --- Bells & Whistles ---
    # 1. Auto Contrast
    contrasted_image = auto_contrast(aligned_image)
    
    # 2. Auto White Balance
    final_image = auto_white_balance(contrasted_image)

    # 3. Auto Cropping 
    h, w, _ = final_image.shape
    crop_h, crop_w = int(h * 0.05), int(w * 0.05)
    cropped_image = final_image[crop_h:-crop_h, crop_w:-crop_w, :]

    # Save the final image
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_aligned.jpg")
    plt.imsave(output_path, cropped_image)
    
    end_time = time.time()
    print(f"  - Saved to {output_path}. Took {end_time - start_time:.2f} seconds.\n")


if __name__ == '__main__':
    data_dir = 'data'
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.tif'))]
    
    for filename in image_files:
        filepath = os.path.join(data_dir, filename)
        process_image(filepath, output_dir)
