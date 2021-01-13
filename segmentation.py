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
    filtered = np.argwhere(img < -930)
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
    # we will search for the first pixel with a HU-value of -900 from the midpoint of the image and update the coordinates accordingly
    extent = img.imageExtent()
    regiongrowing_input = img.getTile((0,0,0,0,0,0),(extent))
    interface.setImage(regiongrowing_input) # output image to interface, so RegionGrowing-module can use it
    x_coordinate, y_coordinate, z_coordinate = int(extent[0]/2), int(0.45 * extent[1]), int(round(0.75 * extent[2], 0))
    print(f"x: {x_coordinate}, y: {y_coordinate}, z: {z_coordinate} for is start point")
    tile2D = img.getTile((0,0,z_coordinate,0,0,0), (extent[0], extent[1])) # missing dimensions will be filled with 1
    print(tile2D)
      
    coordinates = find_nearest_air_coordinates(tile2D, (y_coordinate, x_coordinate))
    coordinates = np.flip(np.concatenate(([z_coordinate], coordinates))) # add z to the vector and flip to mevislab dimension order
    print(f"seed point is {coordinates}")
    seed_vector = [coordinates[0], coordinates[1], coordinates[2], 0, 0, 0] # RegionGrowing module expects a vector of 6 elements
    
    # set default parameters for RegionGrowing Module
    trachea_treshold = -930
    trachea_treshold_step = -64
    ctx.field("RegionGrowing.useAdditionalSeed").value = True
    ctx.field("RegionGrowing.basicNeighborhoodType").value = "BNBH_4D_8_XYZT"
    ctx.field("RegionGrowing.additionalSeed").setVectorValue(seed_vector)
    ctx.field("RegionGrowing.lowerThreshold").value = -2000
    ctx.field("RegionGrowing.upperThreshold").value = trachea_treshold
    ctx.field("RegionGrowing.update").touch() # press update button
    
    trachea_volume = round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)
    trachea_volume_new = trachea_volume
    print(f"{trachea_volume} ml volume initially calculated")
  
    # update treshold to increase volume in a controlled way
    while trachea_treshold_step < 0:
        if trachea_volume_new > 4 * trachea_volume: # check if lung is connected to trachea
            print("lung exploded")
            trachea_treshold = trachea_treshold + trachea_treshold_step # restore old treshold value
            trachea_treshold_step = math.ceil(trachea_treshold_step / 2) # halve the treshold step, if step becomes 0 we are done
            
        trachea_treshold = trachea_treshold - trachea_treshold_step
        
        # if treshold reaches this value, we can assume that the intital RegionGrowing starting from seeding poitnt has failed
        if trachea_treshold > -580: 
            break
        
        
        print(f"new trachea stephold: {trachea_treshold} using a step size of: {trachea_treshold_step} ")
        ctx.field("RegionGrowing.upperThreshold").value = trachea_treshold
        ctx.field("RegionGrowing.update").touch() # press update button
        trachea_volume_new = round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)
        # edge case: final check to break out of loop; region is not twice as large and step is -1
        if ((trachea_treshold_step == -1)  and not (trachea_volume_new > 2 * trachea_volume)):
            print("we're in the edge case")
            break
    print(f'{round(ctx.field("RegionGrowing.segmentedVolume_ml").value, 2)} ml volume after RegionGrowing')
    return ctx.field("RegionGrowing.output0").image()




### -------------- Main ----------- ###
interface = ctx.module("lungs_and_trachea").call("getInterface")
interface1 = ctx.module("RegionGrowingInput").call("getInterface")
interface2 = ctx.module("PythonImage").call("getInterface")
interface3 = ctx.module("PythonImage1").call("getInterface")
interface4 = ctx.module("PythonImage2").call("getInterface")
interface5 = ctx.module("PythonImage3").call("getInterface")
interface6 = ctx.module("PythonImage4").call("getInterface")
interface7 = ctx.module("PythonImage5").call("getInterface")
interface8 = ctx.module("PythonImage6").call("getInterface")


# get the input image field's direct access to the image (which is a MLPagedImageWrapper, see MeVisLab Scripting Reference)
image = ctx.field("input0").image()

# if we have a valid image, continue
if image:
  extent = image.imageExtent()
  
  # create output images that will be filled with data
  lungs_with_trachea = np.empty(reverse(extent), np.int16)
  trachea = np.empty(reverse(extent), np.int16)
  lungs = np.empty(reverse(extent), np.int16)
  img_segmented_mask = np.empty(reverse(extent), np.int16)
  img_trachea_mask = np.empty(reverse(extent), np.int16)
  img_lungs_mask = np.empty(reverse(extent), np.int16)
  trachew_new2 = np.empty(reverse(extent), np.int16)
  lungs_new2 = np.empty(reverse(extent), np.int16)

  #print("Start trachea segmentation")
  #trachea_image = grow_trachea(interface1, image)

  # skimage.measure is needed, but it only supports up to 3 dimensions. Dimension 4 and 6 are 0 in DICOM Data. Dimension 5 are timepoints
  # Loop through each timepoint, selecting tile for that timepoint, shrink and calculate lung segmentation over 3 dimensions. 
  # expand to 6D again after segmentation and merge each timepoint together
  for t in range(0, extent[4]):
    print(f"calculating segmentation over timepoint {t}")
    size   = (extent[0], extent[1], extent[2], extent[3], 1, extent[5])
    
    # Get tiles for timepoint t
    tile6D = image.getTile((0,0,0,0,t,0), size) 
    #trachea6D_mask = trachea_image.getTile((0,0,0,0,t,0), size)
    
    # Make images 3D
    tile3D = tile6D[0, 0, 0, :, :, :]
    #trachea_mask = trachea6D_mask[0, 0, 0, :, :, :]
    
    segmented_mask = segment_thorax(tile3D, True) 
    segmented_tile = segmented_mask * tile3D # * -1 # multiply mask by tile3D to convert binary image to gray-values / HU. Invert hounsfield values
    #selem = ndi.generate_binary_structure(3,2) # structuring element of 3x3x3, creating a ball
    #trachea_mask = ndi.binary_dilation(trachea_mask, selem) # include trachea wand
    #trachea_mask = ndi.binary_closing(trachea_mask, selem) # fill in holes
    #lungs_mask =  segmented_mask - trachea_mask # substract trachea from lungs_trachea to create a mask of lungs only
    #lungs_mask = ndi.binary_opening(lungs_mask, selem) # remove some noise created by the substraction
    #trachea_hu = trachea_mask * segmented_tile # apply trachea mask to segmentation of lungs and trachea
    #lungs_hu = lungs_mask * segmented_tile # apply lungs mask to segmentation of lungs and trachea
    
    # fill in segmented tile into our clean array
    lungs_with_trachea[:, t, :, :, :, :] = make6D(segmented_tile, t) 
    #trachea[:, t, :, :, :, :] = make6D(trachea_hu, t)
    #lungs[:, t, :, :, :, :] = make6D(lungs_hu, t)
    #
    img_segmented_mask[:, t, :, :, :, :] = make6D(segmented_mask, t)
    #img_trachea_mask[:, t, :, :, :, :] = make6D(trachea_mask, t)
    #img_lungs_mask[:, t, :, :, :, :] = make6D(lungs_mask, t)


  # set images to interface
  interface.setImage(lungs_with_trachea)
#  interface1.setImage(lungs_with_trachea) # interface for region growing
  #interface2.setImage(trachea)
  #interface3.setImage(lungs)
  interface4.setImage(img_segmented_mask)
  #interface5.setImage(img_trachea_mask)
  #interface6.setImage(img_lungs_mask)
  
  
#interface = ctx.module("lungs_and_trachea").call("getInterface") # image of segmented thoraqx in HU
#interface1 = ctx.module("RegionGrowingInput").call("getInterface")
#interface2 = ctx.module("PythonImage").call("getInterface")
#interface3 = ctx.module("PythonImage1").call("getInterface")
#interface4 = ctx.module("PythonImage2").call("getInterface")
#interface5 = ctx.module("PythonImage3").call("getInterface")
#interface6 = ctx.module("PythonImage4").call("getInterface")
#interface7 = ctx.module("PythonImage5").call("getInterface")
#interface8 = ctx.module("PythonImage6").call("getInterface")


  
  ## calculate inspiration and expiration timepoints so we can mass correct inspiration images
  #min_tp = ctx.field("CalculateVolume.minTimepoint").value
  #max_tp = ctx.field("CalculateVolume.maxTimepoint").value
  #expiration_volume = []
  #inspiration_volume = []
  #expiration_tp = []
  #inspiration_tp = []
  #print(f"min volume on: {min_tp} and max on: {max_tp}")
  #for t in range(0, extent[4]):
  #  ctx.field("CalculateVolume.userTimepoint").value = t
  #  volume = ctx.field("CalculateVolume.userTimepointVolume").value
  #  print(f"{volume} ml on timepoint {t}")
  # # if ((min_tp < max_tp and (t <= min_tp or t > max_tp)) or (min_tp > max_tp and (t <= min_tp and t > max_tp))):
  #  if (t <= min_tp and t > max_tp):
  #    # determine expiration timepoints. Minimal volume is counted as expiration, max volume is counted as expiration too, acoording to paper of Guerrero
  #    expiration_volume.append(volume)
  #    expiration_tp.append(t)
  #  else:
  #    inspiration_tp.append(t)
  #    inspiration_volume.append(volume)
  #
  #print(f"inspiration on {inspiration_tp} and expiration on {expiration_tp}")
  #
  #discrepency  = (sum(expiration_volume) * len(expiration_tp)) / (sum(inspiration_volume) * len(inspiration_tp))
  #discrepency2  = (sum(expiration_volume) ) / (sum(inspiration_volume))
  #print(f"discrepency is: {discrepency} or {discrepency2}")
  #print(f"expiration {sum(expiration_volume)} over {len(expiration_tp)} timepoints and inspiration {sum(inspiration_volume)} over {len(inspiration_tp)} timepoints")
  ##
  ##
  
  
print("Start trachea segmentation from segmented thorax")
image_new = ctx.field("thorax").image()
thorax_mask = ctx.field("img_segmented_mask").image()

# if we have a valid image, continue
if image_new:
    extent_new = image_new.imageExtent()
    print(f"extent new: {extent_new}")
    trachea_image_new = grow_trachea(interface1, image_new)
#    trachea_image_new = reverse(trachea_image_new)
    
    for t in range(0, extent_new[4]):
        print(t)
        size   = (extent_new[0], extent_new[1], extent_new[2], extent_new[3], 1, extent_new[5])

        # Get tiles for timepoint t of trachea
        trachea6DD = trachea_image_new.getTile((0,0,0,0,t,0), size)
        trachea_mask2 = trachea6DD[0, 0, 0, :, :, :]
        
        # get tiles of segmented thorax
        thorax6D_mask = thorax_mask.getTile((0,0,0,0,t,0), size)
        thorax3D_mask = thorax6D_mask[0, 0, 0, :, :, :]
        
        thorax6D_HU = image_new.getTile((0,0,0,0,t,0), size)
        thorax3D_HU = thorax6D_HU[0, 0, 0, :, :, :]
        
        
        # perform morphological operations on trachea
        selem = ndi.generate_binary_structure(3,2) # structuring element of 3x3x3, creating a ball
        trachea_mask2 = ndi.binary_dilation(trachea_mask2, selem) # include trachea wand
        trachea_mask2 = ndi.binary_closing(trachea_mask2, selem) # fill in holes
        trachea_mask2 = ndi.binary_dilation(trachea_mask2, selem) # include trachea wand thicker


        
        
        lungs_mask_new =  thorax3D_mask - trachea_mask2 # substract trachea from lungs_trachea to create a mask of lungs only
        
        lungs_mask_new = lungs_mask_new > 0 # the subtraction can lead to values of -1, as trachea is removed         from some background. Filter these
        
 #       lungs_mask_new = np.array(lungs_mask_new < -1, dtype=np.int8)
 #       lungs_mask_new = ndi.binary_closing(lungs_mask_new, selem) # remove -1 values because trachea_mask is a bit larger than thorax mask in some places
        lungs_mask_new = ndi.binary_opening(lungs_mask_new, selem) # remove some noise created by the substraction
  #      lungs_mask_new = ndi.binary_opening(lungs_mask_new, selem) # remove some noise created by the substraction
  
        trachea_hu = -1 * trachea_mask2 * thorax3D_HU # apply trachea mask to segmentation of lungs and trachea  
        lungs_hu = -1 * lungs_mask_new * thorax3D_HU # apply lungs mask to segmentation of lungs and trachea

        trachew_new2[:, t, :, :, :, :] = make6D(trachea_mask2, t)
        lungs_new2[:, t, :, :, :, :] = make6D(lungs_mask_new, t)
        trachea[:, t, :, :, :, :] = make6D(trachea_hu, t)
        lungs[:, t, :, :, :, :] = make6D(lungs_hu, t)
#        lungs_mask_new = ndi.binary_opening(lungs_mask_new, selem) # remove some noise created by the substraction
 #   trachea_hu = trachea_mask * segmented_tile # apply trachea mask to segmentation of lungs and trachea
#    lungs_hu = lungs_mask_new * segmented_tile # apply lungs mask to segmentation of lungs and trachea
        
        
    interface2.setImage(trachea)
    interface3.setImage(lungs)    
    
    interface7.setImage(trachew_new2)
    interface8.setImage(lungs_new2)
  
  
    print("segmentation complete")
