import cv2 as cv
import OpenCVUtils
import pydicom as dicom
import scipy.ndimage
from skimage import exposure, measure, segmentation, morphology
from skimage.morphology import disk, ball, opening, closing
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import numpy as np


# Helper function for debugging numpy arrays
def show_array(y):
  print('array: \n', y)
  print('array.ndim: ', y.ndim)
  print('array.shape: ', y.shape)

# Helper function for reversing the order of dimensions from MevisLab to numpy format
def Reverse(tuples): 
    new_tup = tuples[::-1] 
    return new_tup 


# Helper method that returns the label with the maximum volume
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

# Segment lung (-500) / air (-1000) between many other organic tissues
# Filling lung structures removes body tissues that still need to be removed.
# Credits to Guido Zuidhof : https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial#Lung-segmentation
def segment_lung_mask(image, fill_lung_structures=True):
  
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # tissues with a value of > -250 (lungs and air) have a value of 1
    # soft tissues, bonestructure and fluids will have a value of 2.
    binary_image = np.array(image > -250, dtype=np.int8)+1
    labels = measure.label(input = binary_image, background = None,connectivity=2)
    
    # Pick pixel in every corner to determine which label is air.
    # More resistant to "trays" on which the patient lays cutting the air around the person in half
    background_labels = np.unique([labels[0, 0], labels[-1, 0], labels[0, -1], labels[-1, -1]])
    for background_label in background_labels:
      # Fill the air around the person
      binary_image[background_label == labels] = 2
      binary_mask2 = binary_image
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1 # indexing from 0
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 # Make the image actual binary
    binary_image = 1 - binary_image # Invert binary image, lungs are now 1, background is 0
    
    # Remove other air pockets inside the body (noise)
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
        
    return binary_image, binary_mask2

def make6D(image, timepoint):
    image = image[np.newaxis, np.newaxis, np.newaxis, ...] # add back the 3 missing dimensions
    image[..., 2] = 0 # u-dimension
    image[..., 1] = timepoint # time
    image[..., 0] = 0 # c-dimension
    return image




### -------------- Main ----------- ###
interface = ctx.module("PythonImage").call("getInterface")
interface1 = ctx.module("PythonImage1").call("getInterface")
interface2 = ctx.module("PythonImage2").call("getInterface")
# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()

# if we have a valid image, continue
if image:
  extent = image.imageExtent()
  print(f"extent in (x,y,z,c,t,u) order: {extent}")

  # create output image that will be filled in with data
  lungsUnfilled = np.empty(Reverse(extent), np.int16)
  lungsFilled = np.empty(Reverse(extent), np.int16)
  lungsDifference = np.empty(Reverse(extent), np.int16)
  #show_array(lungsUnfilled)
  

  # skimage.measure only supports up to 3 dimensions. Dimension 4 and 6 are 1 in DICOM Data. Dimension 5 are timepoints
  # Loop through each timepoint, select tile for that timepoint, shrink and calculate over 3 dimensions. Join them together and expand to 6D data again
  # WARNING: it seems possible to avoid the loop by applying the 3D mask to 6D image. It seems correct/identical but the voxel properties will be broken
  for t in range(0,extent[4]):
    print(f"calculating over timepoint {t}")
    size   = (extent[0], extent[1], extent[2], extent[3], 1, extent[5])
    tile6D = image.getTile((0,0,0,0,t,0), size) 
    tile3D = tile6D[0, 0, 0, :, :, :]
    
    
    segmented_lungs, binary_mask= segment_lung_mask(tile3D, False) 
    segmented_lungs_filled, binary_mask_filled = segment_lung_mask(tile3D, True) 
    
    # multiply mask by tile3D to convert binary image to greyvalues / HU
    segmented_tile = segmented_lungs * tile3D
    segmented_tile_filled = segmented_lungs_filled * tile3D
    difference = segmented_tile_filled - segmented_tile
    
    # fill in segmented tile into our clean array
    lungsUnfilled[:, t, :, :, :, :] = make6D(segmented_tile, t)
    lungsFilled[:, t, :, :, :, :] = make6D(segmented_tile_filled, t)
    lungsDifference[:, t, :, :, :, :] = make6D(difference, t) 
    #print("Segmented Lungs")
    #show_array(lungsUnfilled)


  # set image to interface
  interface.setImage(lungsUnfilled)
  interface1.setImage(lungsFilled)
  interface2.setImage(lungsDifference)
  print("segmentation complete")
