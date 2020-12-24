import cv2 as cv
import OpenCVUtils
import pydicom as dicom
from scipy import ndimage as ndi
import skimage.segmentation
from skimage import exposure, measure, segmentation, morphology
from skimage.morphology import disk, ball, opening, closing
from skimage.filters import sobel
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import numpy as np
print(f"Numpy version is {np.__version__}")


# Some of the starting Code is taken from ArnavJain, since it's more readable then my own
def generate_markers(image):
    #Creation of the internal Marker
    # The internal marker are definitly lung
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))
    
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


def segment_trachea(image):
  # Trachea will be 1, lungs will be 2
  binary_image = np.array(image > -900, dtype=np.int8)+1
  labels = measure.label(binary_image)

  # Pick the pixel in every corner to determine which label is air.
  # More resistant to "trays" on which the patient lays cutting the air around the person in half
  background_labels = np.unique([labels[0, 0], labels[-1, 0], labels[0, -1], labels[-1, -1]])
  for background_label in background_labels:
    # Fill the air around the person
    binary_image[background_label == labels] = 2
  #We have a lot of remaining small signals outside of the lungs that need to be removed. 
  #In our competition closing is superior to fill_lungs 
  selem = ball(2)
  binary_image = opening(binary_image, selem)
  binary_image -= 1 #Make the image actual binary
  binary_image = 1-binary_image # Invert it, lungs are now 1
  
  return binary_image


# set interface of internal PythonImage module
interface = ctx.module("PythonImage").call("getInterface")
interface1 = ctx.module("PythonImage1").call("getInterface")
interface2 = ctx.module("PythonImage2").call("getInterface")
interface3 = ctx.module("PythonImage3").call("getInterface")
interface4 = ctx.module("PythonImage4").call("getInterface")
interface5 = ctx.module("PythonImage5").call("getInterface")
interface6 = ctx.module("PythonImage6").call("getInterface")
interface7 = ctx.module("PythonImage7").call("getInterface")
# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()


# if we have a valid image, continue
if image:
  extent = image.imageExtent()
  print(f"extent in (x,y,z,c,t,u) order: {extent}")
  
  # create output image that will be modified
  completeTile = image.getTile((0,0,0,0,0,0), image.imageExtent())
  
  # skimage.measure only supports up to 3 dimensions. Dimension 4 and 6 are 1 in DICOM Data. Dimension 5 are timepoints
  # Loop through each timepoint and calculate over 3 dimensions. Join them together
#  for t in range(0,extent[4]):
#  print(f"calculating over timepoint {t}")
  size   = (extent[0], extent[1], extent[2]) # no need to fill all six values, 1 is the default for missing dimensions
  tile = image.getTile((0,0,0), size) # tile is now 3 dimensional
# segmented_lungs = seperate_lungs(tile)
 
  tile2d = image.getTile((0,0), (extent[0], extent[1]))
  
  segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = seperate_lungs(tile2d)
  interface.setImage(segmented)
  interface1.setImage(lungfilter)
  interface2.setImage(outline)
  interface3.setImage(watershed)
  interface4.setImage(sobel_gradient)
  interface5.setImage(marker_internal)
  interface6.setImage(marker_external)
  interface7.setImage(marker_watershed)
  

  #elevation_map = sobel(tile)
  #markers = np.zeros_like(tile, dtype = np.int16)
  #markers[tile < -250] = 1
  #markers[tile > -250] = 2
  #from skimage.segmentation import watershed
  #segmentation = watershed(elevation_map, markers)
  #
  #segmentation = ndi.binary_fill_holes(segmentation - 1)
  #labeled_coins, _ = ndi.label(segmentation)
  #from skimage.color import label2rgb
  #image_label_overlay = label2rgb(labeled_coins, image=tile, bg_label=0)
  
  
  #result = labeled_coins
  
  
  print("Segmentation done")
 
 
 
#  result = segmented_lungs * completeTile

  #    segmented_lungs_fill = segment_lung_mask(tile, True)
  #    difference = segmented_lungs - segmented_morph

  #for t in range(0, extent[4]):
  #  print(f"calculating over timepoint {t}")
  #  size   = (extent[0], extent[1], extent[2]) # no need to fill all six values, 1 is the default for missing dimensions
  #  tile = image.getTile((0,0,0), size) # tile is now 3 dimensional
  #  segmented_trachea = segment_trachea(tile)
  #  result2 = segmented_trachea * completeTile
  #  lungs = result - result2
  # set image to interface
  

