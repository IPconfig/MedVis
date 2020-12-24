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


def make6D(image, timepoint=0):
  """ Converts a 3D image to 6D
  
      Keyword arguments:
      image -- 3D image that needs to be transformed
      timepoint -- time-dimension for this image (default 0)
  """
  image = image[np.newaxis, np.newaxis, np.newaxis, ...] # add back the 3 missing dimensions
  image[..., 2] = 0 # u-dimension
  image[..., 1] = timepoint # time
  image[..., 0] = 0 # c-dimension
  return image




### -------------- Main ----------- ###
interface = ctx.module("PythonImage").call("getInterface")

# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()

# if we have a valid image, continue
if image:
  extent = image.imageExtent()
  print(f"extent in (x,y,z,c,t,u) order: {extent}")

  # create output image that will be filled in with data
  substration_image = np.empty(Reverse(extent), np.int16)

  # skimage.measure only supports up to 3 dimensions. Dimension 4 and 6 are 1 in DICOM Data. Dimension 5 are timepoints
  # Loop through each timepoint, select tile for that timepoint and previous timepoint. 
  # Substract previous timepoint from current timepoint.
  for t in range(0,extent[4]):
    print(f"calculating substraction image for timepoint {t}")
    if t is 0:
      previous = extent[4] - 1 # maximum timepoint
    else:
      previous = t - 1
    print(f"previous timepoint is {previous}")
    
    size   = (extent[0], extent[1], extent[2], extent[3], 1, extent[5]) # only select 1 timepoint
    tile_6D_current = image.getTile((0, 0, 0, 0, t, 0), size)
    tile_6D_previous = image.getTile((0, 0, 0, 0, previous, 0), size)
    tile_3D_current = tile_6D_current[0, 0, 0, :, :, :]
    tile_3D_previous = tile_6D_previous[0, 0, 0, :, :, :]
    substraction = tile_3D_current - tile_3D_previous

    # fill in substraction image into our clean image for timepoint t
    substration_image[:, t, :, :, :, :] = make6D(substraction, t)


  # set image to interface
  interface.setImage(substration_image)

  print("Substraction image is complete")
