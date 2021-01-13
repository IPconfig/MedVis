# -----------------------------------------------------------------------------
## This file contain all functions related to the custom GUI
#
# \file    main.py
# \author  MedVis group 4
# \date    01/2021
#
# -----------------------------------------------------------------------------
from mevis import *

# if modules are not installed, we show an option in the GUI to install dependencies
try:
    import scipy
    import skimage
except:
    pass

def Browse(module): # navigate to path to save segmented images
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
        setCalculateVolumeParameters()
        ReloadModule('SubtractionImage')


def dicomLoaded():
    """ Returns true if dicom files are loaded. Bool used in GUI for conditional shows """
    value = ctx.field("ImportModule.numVolumes").value
    if value > 0:
        ctx.field('dicomLoaded').value = True
    else:
        ctx.field('dicomLoaded').value = False


def StartSegmentation(module):
    """ A new segmentation will need to execute a few actions """
    ReloadModule(module)
    setCalculateVolumeParameters()
    ReloadModule('SubtractionImage')


def saveSegmentation():
    """ Save image of segmented Trachea and image of segmented Lungs in predefined path"""
    ctx.field("SaveTrachea.save").touch()
    ctx.field("SaveLungs.save").touch()
  

def toggleImportSwitches():
    """ Modify switch input of the Trachea and Lungs modules to match the choosen option in the GUI. """
    if ctx.field("segmentationOption").value == "New":
        ctx.field("TracheaSwitch.currentInput").value = 0
        ctx.field("LungsSwitch.currentInput").value = 0
    else:
        ctx.field("TracheaSwitch.currentInput").value = 1
        ctx.field("LungsSwitch.currentInput").value = 1
    setCalculateVolumeParameters()
    ReloadModule('SubtractionImage')


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


def touchButton(module):
    """ Touching a field will cause a notification and the field listener will call the given command. """
    ctx.field(f"{module}").touch()
    
    
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


def calculateDiameter():
    """ Calculates the diameter between two x-marker points """
    if ctx.field("XMarkerListContainer.numItems").value == 2:
        roundedMM = round(ctx.field("PathToKeyFrame.pathLength").value, 2)
        ctx.field("tracheaDiameter").value = roundedMM


def downloadDependencies():
    """ Wrapper to install all dependencies in one click"""
    downloadPipPackage("scipy")
    downloadPipPackage("scikit-image")
    testDependencies()


def downloadPipPackage(package):
    """ Installs a python package using the PythonPip module """
    args = 'install' + ' ' + f'{package}'
    if MLAB.isMacOS():
        args = args + ' ' + '--user'
    ctx.field('PythonPip.command').value = args
    ctx.field('PythonPip.runCommand').touch()
  

def testDependencies():
    """ test if the required libraries are installed in MeVisLab python 
     Show button to install them in GUI if test fails """
    try:
        scipy.__version__
        skimage.__version__
        # dependencies are installed
        ctx.field("dependencyPass").value = True
    except:
        # one or more dependencies are missing
        ctx.field("dependencyPass").value = False
  
  
def changeROItimepoint(field):
    """ Force ROITimepoint to stay in sync """
    if field == "ROIIn.timepoint":
        timepoint = ctx.field("timepointMax").value
    else:
        timepoint = ctx.field("timepointMin").value
        
    if ctx.field(f"{field}").value != timepoint:
        ctx.field(f"{field}").value = timepoint
 
 
def setCalculateVolumeParameters():
    """ Set parameters based on CalculateVolume output """
    updateField('validVolume', 'CalculateVolume.resultsValid')
    updateField('ROIIn.timepoint', 'timepointMax')
    updateField('ROIEx.timepoint', 'timepointMin')


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
    
    # fields from the Trachea diagnoses tab
    updateField('CPRNumMarkers' , 'SoView2DMarkerEditor.numItems')
    updateField('tracheaDiameter', 'PathToKeyFrame.pathLength')
    
    updateField('CPRMarkerColor', 'SoView2DMarkerEditor.color')
    updateField('XMarkerListSort.sortMode', 'CPRMarkerSort')
    updateField('XMarkerListSort.ascending', 'CPRSortAscending')
    updateField('CPRPathLength', 'PathToKeyFrame.pathLength')
    updateField('CPROutputKeys', 'PathToKeyFrame.numOutputKeys')
    updateField('CRPSmoothingField', 'PathToKeyFrame.numSmoothes')
    updateField('CRPResolutionField', 'PathToKeyFrame.outputResolution')
    
    # fields from Lung diagnoses tab
    updateField('timepointVolumeCurrent', 'CalculateVolume.userTimepointVolume')
    updateField('timepointVolumeMin', 'CalculateVolume.minTimepointVolume')
    updateField('timepointVolumeMax', 'CalculateVolume.maxTimepointVolume')
    updateField('timepointCurrent', 'TimepointCounter.currentValue')
    updateField('timepointMin', 'CalculateVolume.minTimepoint')
    updateField('timepointMax', 'CalculateVolume.maxTimepoint')
    updateField('timepointAutoStep', 'TimepointCounter.autoStep')
    updateField('timepointAutoStepInterval', 'TimepointCounter.autoStepInterval_s')
    updateField('timepointStepDirection', 'TimepointCounter.stepDirection')
    
    # Fields used to control GUI conditionals
    setCalculateVolumeParameters()
    
    # check if dicom files are loaded
    dicomLoaded()
    
    # Reset switches to default settings on initialization
    toggleImportSwitches()
    
    # round diameter value if it is set already, so rounded value is available in GUI
    calculateDiameter()
    
    # test if python libraries are installed. show warning + install button in GUI if it isn't
    testDependencies()
    