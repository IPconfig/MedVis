# -----------------------------------------------------------------------------
# This file creates substraction images for DICOM files
#
# \file    LungSegmentation.py
# \author  J.C. de Haan
# \date    12/2020
#
# -----------------------------------------------------------------------------
import numpy as np


def Reverse(tuples): 
  """ Reverses the order of dimensions from MevisLab to numpy format and vice versa. """
  new_tup = tuples[::-1] 
  return new_tup 


def make6D(img, timepoint=0):
  """ Converts a 3D image to 6D
  
      Parameters:
        img: 3D image
        timepoint: time-dimension for this image (default 0)
        
      Returns:
        img: 6D image with given timepoint 
  """
  img = img[np.newaxis, np.newaxis, np.newaxis, ...] # add back the 3 missing dimensions
  img[..., 2] = 0 # u-dimension
  img[..., 1] = timepoint # time
  img[..., 0] = 0 # c-dimension
  return img




### -------------- Main ----------- ###
interface = ctx.module("PythonImage").call("getInterface")

# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()

# if there is a valid image, continue
if image:
  extent = image.imageExtent()
  substration_image = np.empty(Reverse(extent), np.int16) # create blank output image
  
  # failsafe incase there is only 1 timepoint. Return the given image if this happens
  if extent[4] > 1:
    # Loop through each timepoint, select tile for that timepoint and previous timepoint and sub. 
    for t in range(0,extent[4]):
      if t is 0:
        previous = extent[4] - 1 # maximum timepoint
      else:
        previous = t - 1
      # print(f"calculating substraction image for timepoint {t}, substracting image from timepoint {previous}")
      
      size   = (extent[0], extent[1], extent[2], extent[3], 1, extent[5]) # only select 1 timepoint
      tile_6D_current = image.getTile((0, 0, 0, 0, t, 0), size)
      tile_6D_previous = image.getTile((0, 0, 0, 0, previous, 0), size)
      # We cannot subtract 6D tiles as timepoints do not match.
      tile_3D_current = tile_6D_current[0, 0, 0, :, :, :]
      tile_3D_previous = tile_6D_previous[0, 0, 0, :, :, :]
      substraction = tile_3D_current - tile_3D_previous
  
      # fill in substraction image into output image for timepoint t
      substration_image[:, t, :, :, :, :] = make6D(substraction, t)


  # set image to interface
  interface.setImage(substration_image)
  print("Substraction image is calculated")
