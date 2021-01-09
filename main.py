# -----------------------------------------------------------------------------
## This file contain all functions related to the GUI
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
  exp = ctx.expandFilename(ctx.field(f"{module}.name").stringValue())
  if len(exp) == 0:
      exp = ctx.localPath()
  target = MLABFileDialog.getOpenFileName(exp, "Segmented files (*.dcm)", "Open file")
  if target:
    ctx.field(f"{module}.name").value = target
    ctx.field(f"{module}.load").touch()
  


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
    print(f"Touched field: {Field}")
    Field.touch()
    
    
def ReloadModule(module):
    """ Reload a MeVisLab module with a given name. """
    ctx.module(f"{module}").reload()
    print(f"reloaded module")


def updateField(target, source):
    """ Updates a target field with the value of the source field. """
    ctx.field(f"{target}").value = ctx.field(f"{source}").value
    #print(f"Updated {target} to value of {source}")


def updateCounter(changedField):
  ''' If the slider in the GUI changes, push the change to the counter in Mevislab. '''
  ctx.field("TimepointCounter.currentValue").value = changedField.value


def updateSliderField(changedField):
  ''' If MevisLab Counter changes, push the change to the slider in the GUI. '''
  ctx.field("TimepointCurrent").value = changedField.value


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
    updateField('timePointVolume', 'CalculateVolume.userTimepointVolume')
    updateField('validVolume', 'CalculateVolume.resultsValid')
    
    
    # Reset switches to default settings on initialization
    toggleImportSwitches()