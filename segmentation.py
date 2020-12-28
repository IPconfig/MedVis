# -----------------------------------------------------------------------------
## This file implements segmentation from lungs and trachea from DICOM Torax files
#
# \file    LungSegmentation.py
# \author  J.C. de Haan
# \date    12/2020
#
# -----------------------------------------------------------------------------



# import mevis
#import cv2
#import OpenCVUtils
import numpy as np
from skimage import measure
from skimage.morphology import ball, binary_opening, binary_closing, binary_dilation
import math

# ----------- Helper functions ----------- #
# Helper function for debugging numpy arrays
def show_array(y):
  #print('array: \n', y)
  print('array.ndim: ', y.ndim)
  print('array.shape: ', y.shape)

# Helper function for reversing the order of dimensions from MevisLab to numpy format
def reverse(tuples): 
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
def segment_lung_and_trachea_mask(img, fill_lung_structures=True):
  
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # tissues with a value of > -250 (lungs and air) have a value of 1
    # soft tissues, bonestructure and fluids will have a value of 2.
    binary_image = np.array(img > -250, dtype=np.int16)+1
    labels = measure.label(input = binary_image, background = None,connectivity=3)
    
    # Pick pixel in every corner to determine which label is air.
    # More resistant to "trays" on which the patient lays cutting the air around the person in half  
    background_labels = np.unique([labels[0, 0], labels[-1, 0], labels[0, -1], labels[-1, -1]])
    for background_label in background_labels:
      # Fill the air around the person
      binary_image[labels == background_label] = 2
    
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
        
    return binary_image

def make6D(img, timepoint):
    img = img[np.newaxis, np.newaxis, np.newaxis, ...] # add back the 3 missing dimensions
    img[..., 2] = 0 # u-dimension
    img[..., 1] = timepoint # time
    img[..., 0] = 0 # c-dimension
    return img
  
# Helper method that calculates the euclidian distance of a (filtered) image to the given coordinates.
# Returns the coordinates closest by the target coordinates
def find_nearest_air_coordinates(img, coordinates):
    filtered = np.argwhere(img < -900)
    distances = np.sqrt((filtered[:,0] - coordinates[0]) ** 2 + (filtered[:,1] - coordinates[1]) ** 2)
    nearest_index = np.argmin(distances)
    return filtered[nearest_index]
  
def grow_trachea(interface, img, extent):
  # we need to find a seed point of the trachea for the regiongrowing module.
  # at around 75% on the z-axis, the trachea is located a little left and above of the midpoint
  # we will search for the first pixel with a HU-value of -900 from the midpoint of the image and update the coordinates accordingly
  regiongrowing_input = img.getTile((0,0,0,0,0,0),(extent))
  interface.setImage(regiongrowing_input) # output image to interface, so RegionGrowing-module can use it
  x_coordinate, y_coordinate, z_coordinate = int(extent[0]/2), int(extent[1]/2), int(round(0.75 * extent[2], 0))
  #print(f"x: {x_coordinate}, y: {y_coordinate}, z: {z_coordinate} for slice that trachea uses")
  tile2D = img.getTile((0,0,z_coordinate,0,0,0), (extent[0], extent[1])) # missing dimensions will be filled with 1

  coordinates = (y_coordinate,x_coordinate)
  coordinates = find_nearest_air_coordinates(tile2D, (x_coordinate, y_coordinate))
  coordinates = np.flip(np.concatenate(([z_coordinate], coordinates))) # add z to the vector and flip to mevislab dimension order
  seed_vector = [coordinates[0], coordinates[1], coordinates[2], 0, 0, 0] # RegionGrowing module expects a vector of 6 elements
  
  trachea_treshold = -900
  trachea_treshold_step = -64
  ctx.field("RegionGrowing.useAdditionalSeed").value = True
  ctx.field("RegionGrowing.basicNeighborhoodType").value = "BNBH_4D_8_XYZT"
  ctx.field("RegionGrowing.additionalSeed").setVectorValue(seed_vector)
  ctx.field("RegionGrowing.lowerThreshold").value = -2000
  ctx.field("RegionGrowing.upperThreshold").value = trachea_treshold
  ctx.field("RegionGrowing.update").touch() # press update button
  trachea_volume = round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)
  trachea_volume_new = trachea_volume
  print(f"{trachea_volume} ml volume calculated")

  while trachea_treshold_step <= -1:
    if trachea_volume_new > 2 * trachea_volume: # check if lung is connected to trachea     
      trachea_treshold = trachea_treshold + trachea_treshold_step # restore old treshold value
      trachea_treshold_step = math.floor(trachea_treshold_step / 2) # halve the treshold step
    
    trachea_treshold = trachea_treshold - trachea_treshold_step
    ctx.field("RegionGrowing.upperThreshold").value = trachea_treshold
    ctx.field("RegionGrowing.update").touch() # press update button
    trachea_volume_new = round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)

    # if step is still too large, decrease step by 1 and break. This is guaranteed to be the largest possible region without including a lung
    if trachea_treshold_step == -1 and trachea_volume_new > 2 * trachea_volume:
      trachea_treshold = trachea_treshold - 1
      ctx.field("RegionGrowing.upperThreshold").value = trachea_treshold
      ctx.field("RegionGrowing.update").touch() # press update button
      break
  print(f'{round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)} ml volume final')
  return ctx.field("RegionGrowing.output0").image()




### -------------- Main ----------- ###
interface = ctx.module("lungs_and_trachea").call("getInterface")
interface1 = ctx.module("RegionGrowingInput").call("getInterface")
interface2 = ctx.module("PythonImage").call("getInterface")
interface3 = ctx.module("PythonImage1").call("getInterface")

# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()

# if we have a valid image, continue
if image:
  extent = image.imageExtent()
  print(f"extent in (x,y,z,c,t,u) order: {extent}")

  
  # create output images that will be filled in with data
  lungs_with_trachea = np.empty(reverse(extent), np.int16)
  trachea = np.empty(reverse(extent), np.int16)
  lungs = np.empty(reverse(extent), np.int16)

  # apply RegionGrowing module to grow the trachea without connecting to lung
  trachea_image = grow_trachea(interface1, image, extent)
  

  # skimage.measure is needed, but it only supports up to 3 dimensions. Dimension 4 and 6 are 0 in DICOM Data. Dimension 5 are timepoints
  # Loop through each timepoint, selecting tile for that timepoint, shrink and calculate lung segmentation over 3 dimensions. 
  # expand to 6D again after segmentation and merge each timepoint together
  for t in range(0, extent[4]):
    print(f"calculating segmentation over timepoint {t}")
    size   = (extent[0], extent[1], extent[2], extent[3], 1, extent[5])
    
    # Get tiles for timepoint t
    tile6D = image.getTile((0,0,0,0,t,0), size) 
    trachea6D_mask = trachea_image.getTile((0,0,0,0,t,0), size)
    
    # Make images 3D
    tile3D = tile6D[0, 0, 0, :, :, :]
    trachea_mask = trachea6D_mask[0, 0, 0, :, :, :]
    
    segmented_mask = segment_lung_and_trachea_mask(tile3D, True) 
    segmented_tile = segmented_mask * tile3D * -1 # multiply mask by tile3D to convert binary image to gray-values / HU. Invert hounsfield values
    selem = ball(3) # structuring element of 3x3x3
    trachea_mask = binary_dilation(trachea_mask, selem) # include trachea wand
    trachea_mask = binary_closing(trachea_mask, selem) # fill in holes
    lungs_mask =  segmented_mask - trachea_mask # substract trachea from lungs_trachea to create a mask of lungs only
    lungs_mask = binary_opening(lungs_mask, selem) # remove some noise created by the substraction

    trachea_hu = trachea_mask * segmented_tile # apply trachea mask to segmentation of lungs and trachea
    lungs_hu = lungs_mask * segmented_tile # apply lungs mask to segmentation of lungs and trachea
    
    # fill in segmented tile into our clean array
    lungs_with_trachea[:, t, :, :, :, :] = make6D(segmented_tile, t) 
    trachea[:, t, :, :, :, :] = make6D(trachea_hu, t)
    lungs[:, t, :, :, :, :] = make6D(lungs_hu, t)


  # set images to interface
  interface.setImage(lungs_with_trachea)
  interface2.setImage(trachea)
  interface3.setImage(lungs)
  print("segmentation complete")
