# -----------------------------------------------------------------------------
## This file contain all functions related to the custom GUI
#
# \file    main.py
# \author  MedVis group 4
# \date    01/2021
#
# -----------------------------------------------------------------------------
from mevis import *

def Browse(module):
    """ Browse button opens a file dialog. """
    exp = ctx.expandFilename(ctx.field(f"{module}").stringValue())
    if len(exp) == 0:
      exp = ctx.localPath()
    target = MLABFileDialog.getExistingDirectory(exp, "Open directory", MLABFileDialog.ShowDirsOnly)
    if target:
        ctx.field(f"{module}").value = target
        if module == "SaveTrachea.sourceName": # if path for SaveTrachea is updated, also update path for lungs
            ctx.field("SaveLungs.sourceName").value = target
            
            
def BrowseFile(module):
    """ Browse dialog used to open a file. It also checks if the file is valid for GUI purposes"""
  exp = ctx.expandFilename(ctx.field(f"{module}.name").stringValue())
  if len(exp) == 0:
      exp = ctx.localPath()
  target = MLABFileDialog.getOpenFileName(exp, "Segmented files (*.dcm)", "Open file")
  if target:
    ctx.field(f"{module}.name").value = target
    ctx.field(f"{module}.load").touch()
    # check if result is valid when you switch inputs. Field used for conditional GUI
    updateField('validVolume', 'CalculateVolume.resultsValid')


def StartSegmentation(module):
    """ A new segmentation will need to execute a few actions """
    ReloadModule(module)
    updateField('validVolume', 'CalculateVolume.resultsValid')


def saveSegmentation():
    """ Save image of segmented Trachea and image of segmented Lungs in predefined path"""
    ctx.field("SaveTrachea.save").touch()
    ctx.field("SaveLungs.save").touch()
  

def toggleImportSwitches():
    """ Modify switch input of the Trachea and Lungs modules to match the choosen option in the GUI. """
    if ctx.field("segmentationOption").value == "New":
        ctx.field("TracheaSwitch.currentInput").value = 0
        ctx.field("LungsSwitch.currentInput").value = 0
        # check if result is valid when you switch inputs. Field used for conditional GUI
        updateField('validVolume', 'CalculateVolume.resultsValid')
    else:
        ctx.field("TracheaSwitch.currentInput").value = 1
        ctx.field("LungsSwitch.currentInput").value = 1
        updateField('validVolume', 'CalculateVolume.resultsValid')


def toggleSampling():
    """ Modify switch input of the SampleToggle module to match the choosen resampling option in the GUI. """
    if ctx.field("SamplingDICOM").value == "Downsampling" :
        ctx.field("SampleToggle.currentInput").value = 0
    elif ctx.field("SamplingDICOM").value == "Upsampling":
        ctx.field("SampleToggle.currentInput").value = 2
    else:
      ctx.field("SampleToggle.currentInput").value = 1


def updateResample(name):
    """ Updates x,y,z parameters in the up/downsampling modules to match the GUI values. """
    updateField(f"DownSampling.{name}Resample", f"{name}Resample")
    updateField(f"UpSampling.{name}Resample", f"{name}Resample")


def touchField(Field):
    """ Touching a field will cause a notification and the field listener will call the given command. """
    Field.touch()
    
    
def ReloadModule(module):
    """ Reload a MeVisLab module with a given name. """
    ctx.module(f"{module}").reload()
    
    
def updateField(target, source):
    """ Updates a target field with the value of the source field. """
    ctx.field(f"{target}").value = ctx.field(f"{source}").value
    #print(f"Updated {target} to value of {source}")

def clearField(target):
    ctx.field(f"{target}").value = None

def updateCounter(changedField):
    """ If the slider in the GUI changes, push the change to the counter in Mevislab. """
    ctx.field("TimepointCounter.currentValue").value = changedField.value


def updateSliderField(changedField):
    """ If MevisLab Counter changes, push the change to the slider in the GUI. """
    ctx.field("timepointCurrent").value = changedField.value



## CPR 
# Delete All Markers in View2D
def Delete(target):
  ctx.field(f"{target}").touch()
  

def calculateDiameter():
    """ Calculates the diameter between two x-marker points """
    if ctx.field("XMarkerListContainer.numItems").value == 2:
      ctx.field("tracheaDiameter").value = round(ctx.field("PathToKeyFrame1.pathLength").value, 2)


# Print diameter between selected markers
def printstr(SomeString):
  print(f"{SomeString} ")
  image = round(ctx.field("PathToKeyFrame1.pathLength").value,1)
  print(f"{image} mm")
  
  

def initialize():
    """
      Initialize all field values used in the control panels. This fills up our parameter fields with the values from the network
    """
    updateField('dataPath', 'ImportModule.fullPath')
    updateField('SegmentationPath', 'SaveTrachea.sourceName')
    updateField('TracheaPathFileName', 'ProcessedTrachea.name')
    updateField('LungPathFileName', 'ProcessedLungs.name')
    
    updateField('xResample', 'DownSampling.xResample')
    updateField('yResample', 'DownSampling.yResample')
    updateField('zResample', 'DownSampling.zResample')
    updateField('zResample', 'DownSampling.zResample')
    
    updateField('CPRNumMarkers' , 'SoView2DMarkerEditor.numItems')
    updateField('tracheaDiameter', 'PathToKeyFrame1.pathLength')
    
    updateField('validVolume', 'CalculateVolume.resultsValid')
    updateField('timepointVolumeCurrent', 'CalculateVolume.userTimepointVolume')
    updateField('timepointVolumeMin', 'CalculateVolume.minTimepointVolume')
    updateField('timepointVolumeMax', 'CalculateVolume.maxTimepointVolume')
    
    updateField('timepointCurrent', 'TimepointCounter.currentValue')
    updateField('timepointMin', 'CalculateVolume.minTimepoint')
    updateField('timepointMax', 'CalculateVolume.maxTimepoint')
    updateField('timepointAutoStep', 'TimepointCounter.autoStep')
    updateField('timepointAutoStepInterval', 'TimepointCounter.autoStepInterval_s')
    updateField('timepointStepDirection', 'TimepointCounter.stepDirection')
    
    
    
    
    # Reset switches to default settings on initialization
    toggleImportSwitches()
    