import cv2 as cv
import OpenCVUtils
import pydicom as dicom
import scipy.ndimage
from skimage import exposure, measure
from skimage.morphology import disk, opening, closing
import numpy as np
print(f"Numpy version is {np.__version__}")


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
  
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # tissues with a value of > -320 (lungs and air) have a value of 1, other tissues and water will have a value of 2.
    binary_image = np.array(image > -320, dtype=np.int16)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
        
    return binary_image


interface = ctx.module("PythonImage").call("getInterface")
# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()
# if we have a valid image, continue
if image:
  extent = image.imageExtent()
  print(f"extent contains the following: {extent}")
  # mevislab expect (x,y,z,c,t,u)
  
  # create output image that will be modified
  completeTile = image.getTile((0,0,0,0,0,0), image.imageExtent())
  
  # skimage.measure only supports up to 3 dimensions. Dimension 4 and 6 are 1 in DICOM Data. Dimension 5 are timepoints
  # Loop through each timepoint and calculate over 3 dimensions. Join them together
  for t in range(0,extent[4]):
  #  print(f"calculating over timepoint {t}")
    size   = (extent[0], extent[1], extent[2]) # no need to fill all six values, 1 is the default for missing dimensions
    tile = image.getTile((0,0,0), size) # tile is now 3 dimensional
    
    
  #  segmented_lungs = segment_lung_mask(tile, False)
    segmented_lungs_fill = segment_lung_mask(tile, True)
  #  difference = segmented_lungs_fill - segmented_lungs

    result = segmented_lungs_fill * completeTile
      
  # set image to interface
  interface.setImage(result, minMaxValues = (image.minVoxelValue(), image.maxVoxelValue()),
                             voxelToWorldMatrix = image.voxelToWorldMatrix())
