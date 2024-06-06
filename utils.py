import czifile
import numpy as np

# Input image loader (.czi files)
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