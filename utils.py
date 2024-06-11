import czifile
import numpy as np
from tensorflow.python.client import device_lib
import pandas as pd
from skimage import measure

# Function to get GPU details
def get_gpu_details():
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == 'GPU':
            print(f"Device name: {device.name}")
            print(f"Device type: {device.device_type}")
            print(f"GPU model: {device.physical_device_desc}")

# Input image loader and channel reorg (.czi files)
def read_czi_image(filename, channels):
    """Image loader for .czi files, removes singleton dimension and rearranges array axes from ZXY to XYZ.
    This step ensures that the input data conforms to the expected input shape for the UNET model"""

    # Read .czi file
    x = czifile.imread(filename) 
    # Remove singleton dimensions in the last axis
    x = np.squeeze(x, axis=3) 
    # Rearrange Z axis from first (source, 0) to last (destination, -1) position
    # Tensorflow expects a "channels-last" array (height, width, channels)
    x = np.moveaxis(x, 0, -1) 
    # Select channels to keep based on the channels tuple argument
    x = x[:, :, channels]
    # Returns a numpy.ndarray of dtype "uint8"
    return x

def read_czi_for_napari(filename):
    """Image loader for .czi files, removes singleton dimension"""

    # Read .czi file
    x = czifile.imread(filename) 
    # Remove singleton dimensions in the last axis
    x = np.squeeze(x, axis=3) 
    # Returns a numpy.ndarray of dtype "uint8"
    return x

def check_overlap(r1,r2,r3):
    '''Checks overlaps in between all 3 different class labels using a numpy logical operator
    and outputs a boolean'''
    print (f"\nBackground and Healthy share pixels: {str(np.logical_and(r1,r2).any())}")
    print (f"Background and Tumor share pixels: {str(np.logical_and(r1,r3).any())}")
    print (f"Healthy and Tumor share pixels: {str(np.logical_and(r2,r3).any())}")


def fix_overlap(raw_masks, f_masks):
    '''Iterates over the masks(arrays) in groups of 3 (background, healthy, tumor), finds shared positions
    among them and substitutes them for 0s in the original array, finally it appends the modified arrays
    into a masks list'''
    index = 0 #Just an index to keep track of the sample we are processing the labels for

    masks = []
    bh_shared = []
    bt_shared = []
    ht_shared = []
    
    for r1, r2, r3 in zip(raw_masks[::3], raw_masks[1::3], raw_masks[2::3]):
        
        # -------------------------
        # Check if r1 and r2 share positive values at the same position (background and healthy)
        if np.logical_and(r1,r2).any():
            shared_positions = np.logical_and(r1 > 0, r2 > 0)
            # Modify positions to have a value of 0 in r1 and r2
            r1[shared_positions] = 0
            r2[shared_positions] = 0
            # Generate r4 with a value of 255 in shared positions
            r4 = np.zeros_like(r1)
            r4[shared_positions] = 255
        else:
            r4 = np.zeros_like(r1)
        # Append r4 to bh_shared
        bh_shared.append(r4)
        
        # -------------------------
        # Check if r1 and r3 share positive values at the same position (background and healthy)
        if np.logical_and(r1,r3).any():
            shared_positions = np.logical_and(r1 > 0, r3 > 0)
            # Modify positions to have a value of 0 in r1 and r2
            r1[shared_positions] = 0
            r3[shared_positions] = 0
            # Generate r4 with a value of 255 in shared positions
            r4 = np.zeros_like(r1)
            r4[shared_positions] = 255
        else:
            r4 = np.zeros_like(r1)
        # Append r4 to bh_shared
        bt_shared.append(r4)
        
        # -------------------------
        # Check if r2 and r3 share positive values at the same position (background and healthy)
        if np.logical_and(r2,r3).any():
            shared_positions = np.logical_and(r2 > 0, r3 > 0)
            # Modify positions to have a value of 0 in r1 and r2
            r2[shared_positions] = 0
            r3[shared_positions] = 0
            # Generate r4 with a value of 255 in shared positions
            r4 = np.zeros_like(r1)
            r4[shared_positions] = 255
        else:
            r4 = np.zeros_like(r1)
        # Append r4 to bh_shared
        ht_shared.append(r4)

        # --------------------------
        # Displays the results of fixing the overlapping pixel values
        print("\nFixing overlap results: \n")
        print(f_masks[index].name.replace('_Background.ome.tiff', ''))
        check_overlap(r1,r2,r3) #Checks if different class labels overlap

        # --------------------------
        # Adds the new arrays without overlapping pixels between classes into a list
        u = np.stack((r1>0, r2>0, r3>0), -1).astype(np.float32) # Create a numpy.ndarray of shape (px height, px width, 3)
        print(f"Output mask array: {u.shape} \n")
        masks.append(u) # Appends the array to the masks list
        
        index += 3 # Update the index to keep track of the sample position
                
    return masks, bh_shared, bt_shared, ht_shared

# Calculate the overall mean intensity of pixels within the combined mask of all labels
# Using the weighted average of intensity_mean weighted by area
# This approach effectively simulates merging all labels into one mask and then calculating the mean intensity
# It performs some extra computations but I can reuse this function to extract per labels stats later on

# Transform extract weighted_intensity_mean into a function
def extract_weighted_mean_intensity (mask_input, intensity_input):

    labels = measure.label(mask_input)

    label_props = measure.regionprops_table(label_image=labels, intensity_image=intensity_input, properties=["label","area","intensity_mean"])

    df = pd.DataFrame(label_props)

    # Calculate the total area occupied by healthy labels
    total_area = df['area'].sum()

    weighted_intensity_mean = (df['intensity_mean'] * df['area']).sum() / total_area

    return weighted_intensity_mean

# You can also calculate the overall mean intensity of pixels within the combined mask of all labels directly using a vectorized approach

def extract_mean_intensity(mask_input, intensity_input):
    # Create a boolean mask where labels are greater than zero
    mask = mask_input > 0
    
    # Extract the relevant pixels from the intensity image using the mask
    masked_intensity_values = intensity_input[mask]
    
    # Compute the mean intensity of these masked pixels
    intensity_mean = np.mean(masked_intensity_values)
    
    return intensity_mean

def calculate_healthy_tumor_percentage(predicted_classes):

    # Extract tumor and background classes as separate arrays
    tumor_class = predicted_classes == 2
    healthy_class = predicted_classes == 1

    # Calculate the total number of pixels
    total_pixels = predicted_classes.size

    # Calculate the number of pixels occupied by tumor and healthy classes
    tumor_pixels = np.sum(tumor_class)
    healthy_pixels = np.sum(healthy_class)

    # Calculate the number of pixels occupied by the tissue (healthy + tumor)
    tissue_pixels = tumor_pixels + healthy_pixels

    # Calculate the percentage of total tissue area occupied by each class
    tumor_percentage = (tumor_pixels / tissue_pixels) * 100
    healthy_percentage = (healthy_pixels / tissue_pixels) * 100

    return tumor_percentage, healthy_percentage