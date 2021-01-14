# -----------------------------------------------------------------------------
# This file implements segmentation from lungs and trachea from DICOM Torax files
#
# \file    LungSegmentation.py
# \author  J.C. de Haan
# \date    12/2020
#
# -----------------------------------------------------------------------------
import numpy as np
import math

# if modules are not installed, we show an option in the GUI to install dependencies. 
# For a better user experience, we don't want to log these errors
try: 
    from skimage import measure
    from scipy import ndimage as ndi
except:
    print("Please install 'scikit-image' and 'scipy' using the PythonPip module or via the GUI before using the segmentation module.")


# ----------- Helper functions ----------- #
def reverse(tuples): 
    """ Reverses the order of dimensions from MevisLab to numpy format and vice versa. """
    new_tup = tuples[::-1]
    return new_tup 


def largest_label_volume(im, bg=-1):
    """ Helper function for segment_thorax() that returns the label with the maximum volume. """
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


# Inspired by Guido Zuidhof: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial#Lung-segmentation
def segment_thorax(img, fill_lung_structures=True):
    """
      Segment lung (-500) / air (-1000) between many other organic tissues
      
      Parameters:
        img: 6D DICOM image
        fill_lung_structures: Set to True if you want to include lung structures that fell outside the treshold (default: True)
        
      Returns:
        binary_image: mask of segmented lungs, bronchi and trachea
    """
  
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # tissues with a value of > -250 (lungs and air) have a value of 1
    # soft tissues, bonestructure and fluids will have a value of 2.
    binary_image = np.array(img > -250, dtype=np.int8)+1
    labels = measure.label(input = binary_image, background = None,connectivity=3)
    
    # Pick pixel in every corner to determine which label is air.
    # More resistant to "trays" on which the patient lays cutting the air around the person in half  
    background_labels = np.unique([labels[0, 0], labels[-1, 0], labels[0, -1], labels[-1, -1]])
    for background_label in background_labels:
      # Fill the air around the person
      binary_image[labels == background_label] = 2
    
    # Method of filling the lung structures (that is superior to something like morphological closing)
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
  

def find_nearest_air_coordinates(img, coordinates):
    """
      Helper method that calculates the euclidian distance of a (filtered) image to the given coordinates.
      
      Parameters:
        img:       2D image
      
      Returns:
        (x, y): coordinates closest by the target coordinates
    """
    filtered = np.argwhere(img > 920)
    distances = np.sqrt((filtered[:,0] - coordinates[0]) ** 2 + (filtered[:,1] - coordinates[1]) ** 2)
    nearest_index = np.argmin(distances)
    return filtered[nearest_index]
 
 
def grow_trachea(interface, img):
    """
        Calculate a seed point at 95% on the the z-axis
        Apply regionGrowing module starting from calculated seed point until the volume explodes
        
        Parameters:
          interface: Interface of the regionGrowing module
          img:       6D input image
        
        Returns:
          img: Segmented image of the trachea
    """
    # To prevent some slices not containing trachea, we start the seed-point from 95% along the z-axis
    # we will search for the first pixel with a HU-value of -920 from the midpoint of the image and update the coordinates accordingly
    extent = img.imageExtent()
    regiongrowing_input = img.getTile((0,0,0,0,0,0),(extent))
    interface.setImage(regiongrowing_input) # output image to interface, so RegionGrowing-module can use it
    x_coordinate, y_coordinate, z_coordinate = int(0.5 * extent[0]), int(0.5 * extent[1]), int(0.95 * extent[2])
    tile2D = img.getTile((0,0,z_coordinate,0,0,0), (extent[0], extent[1])) # missing dimensions will be filled with 1
    
    coordinates = find_nearest_air_coordinates(tile2D, (y_coordinate, x_coordinate))
    coordinates = np.flip(np.concatenate(([z_coordinate], coordinates))) # add z to the coordinates and flip to mevislab dimension order
    seed_vector = [coordinates[0], coordinates[1], coordinates[2], 0, 0, 0] # RegionGrowing module expects a vector of 6 elements
    
    # set default parameters for RegionGrowing Module
    trachea_treshold = 920
    trachea_treshold_step = 64
    ctx.field("RegionGrowing.useAdditionalSeed").value = True
    ctx.field("RegionGrowing.basicNeighborhoodType").value = "BNBH_4D_8_XYZT"
    ctx.field("RegionGrowing.additionalSeed").setVectorValue(seed_vector)
    ctx.field("RegionGrowing.lowerThreshold").value = trachea_treshold
    ctx.field("RegionGrowing.upperThreshold").value = 2000
    ctx.field("RegionGrowing.update").touch()
    
    trachea_volume = round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)
    trachea_volume_new = trachea_volume
    print(f"{trachea_volume} ml volume initially calculated")
  
    # update treshold to increase volume in a controlled way by checking if lung is connected to trachea
    while trachea_treshold_step > 0:
        if trachea_volume_new > 2 * trachea_volume: 
            print("lung exploded")
            trachea_treshold = trachea_treshold + trachea_treshold_step # restore old treshold value
            trachea_treshold_step = math.floor(trachea_treshold_step / 2) # halve the treshold step
            
        trachea_treshold = trachea_treshold - trachea_treshold_step
        if trachea_treshold < 600: 
            break # if treshold reaches this value, we can assume the seed marker is out of bounds

        ctx.field("RegionGrowing.lowerThreshold").value = trachea_treshold
        ctx.field("RegionGrowing.update").touch() # press update button
        trachea_volume_new = round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)
        
    print(f'{round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)} ml volume after RegionGrowing')
    return ctx.field("RegionGrowing.output0").image()





### -------------- Main ----------- ###
interfaceRegionGrowing = ctx.module("RegionGrowingInput").call("getInterface")
interface1 = ctx.module("thorax").call("getInterface")
interface2 = ctx.module("thorax_mask").call("getInterface")
interface3 = ctx.module("trachea").call("getInterface")
interface4 = ctx.module("lungs").call("getInterface")


# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()

if image:
  extent = image.imageExtent()
  
  # create output images that will be filled with data
  thorax = np.empty(reverse(extent), np.int16)
  thorax_mask = np.empty(reverse(extent), np.int8)
    
  # skimage.measure is needed to start segmentation, but it only supports up to 3 dimensions. 
  # Dimension 4 and 6 are 0 in DICOM Data. Dimension 5 are timepoints.
  # Retrieve 3D tile for each timepoint and create segmentation
  # expand to 6D again and merge each timepoint together
  for t in range(0, extent[4]):
    print(f"calculating thorax segmentation over timepoint {t}")
    size   = (extent[0], extent[1], extent[2], extent[3], 1, extent[5])
    tile6D = image.getTile((0,0,0,0,t,0), size) # Get tiles for timepoint t
    tile3D = tile6D[0, 0, 0, :, :, :]
    segmented_thorax_mask = segment_thorax(tile3D, True) # Include dead space
    
    # Apply mask, making HU positive so MeVislab can calculate volume of lungs correctly
    segmented_thorax = np.abs(np.where(segmented_thorax_mask, tile3D, 0))
    
    # fill in segmented tile into our clean array
    thorax[:, t, :, :, :, :] = make6D(segmented_thorax, t) 
    thorax_mask[:, t, :, :, :, :] = make6D(segmented_thorax_mask, t)
    
  # Set as MevisLab image
  interface1.setImage(thorax)
  interface2.setImage(thorax_mask)



thorax_image = ctx.field("thorax").image()
thorax_mask = ctx.field("thorax_mask").image()

if thorax_image:
    print("Start trachea segmentation from segmented thorax")
    extent = thorax_image.imageExtent()
    print(f"extent new: {extent}")
    # create output images that will be filled with data
    trachea = np.empty(reverse(extent), np.int16)
    lungs = np.empty(reverse(extent), np.int16)
    trachea_mask = np.empty(reverse(extent), np.int8)
    lungs_mask = np.empty(reverse(extent), np.int8)

    # Segmentation of trachea in 4D, output a mask
    trachea_image = grow_trachea(interfaceRegionGrowing, thorax_image)
    
    for t in range(0, extent[4]):
        print(f"Segmentation of Trachea and Lungs for timepoint {t}")
        size = (extent[0], extent[1], extent[2], extent[3], 1, extent[5])
        # Get tiles for timepoint t of trachea mask, thorax mask and thorax HU
        trachea6D_mask = trachea_image.getTile((0,0,0,0,t,0), size)
        trachea3D_mask = trachea6D_mask[0, 0, 0, :, :, :]
        thorax6D_mask = thorax_mask.getTile((0,0,0,0,t,0), size)
        thorax3D_mask = thorax6D_mask[0, 0, 0, :, :, :]
        thorax6D_HU = thorax_image.getTile((0,0,0,0,t,0), size)
        thorax3D_HU = thorax6D_HU[0, 0, 0, :, :, :]
        
        # perform morphological operations on trachea
        selem = ndi.generate_binary_structure(3,2) # structuring element of 3x3x3, creating a ball
        trachea3D_mask = ndi.binary_dilation(trachea3D_mask, selem) # include trachea wand
        trachea3D_mask = ndi.binary_closing(trachea3D_mask, selem) # fill in holes
        trachea3D_mask = ndi.binary_dilation(trachea3D_mask, selem) # make trachea wand thicker

        # Create lung mask
        lungs3D_mask =  thorax3D_mask - trachea3D_mask
        lungs3D_mask = lungs3D_mask > 0 # filter values of -1; created by subtracting trachea from background
        lungs3D_mask = ndi.binary_opening(lungs3D_mask, selem) # remove some noise created by the substraction

        # Apply mask to HU
        trachea_hu =  np.where(trachea3D_mask, thorax3D_HU, 0)
        lungs_hu = np.where(lungs3D_mask, thorax3D_HU, 0)
        
        # fill in segmented tile into our clean array
        trachea[:, t, :, :, :, :] = make6D(trachea_hu, t)
        lungs[:, t, :, :, :, :] = make6D(lungs_hu, t)
     
    # Set as MevisLab image  
    interface3.setImage(trachea)   
    interface4.setImage(lungs)
  
  
    print("segmentation complete")
