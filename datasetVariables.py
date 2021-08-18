from pathlib import Path
from utilities import *

rawDataDir = "Raw_Data"

sequenceLen = 200
dimensionsCount = 9
datasetFile = Path("Data_Set/normalized_data_set.pickle")
if(datasetFile.is_file()):
    dataSet = loadData("normalized_data_set")    
    sequenceLen = dataSet.shape[1]
    dimensionsCount = dataSet.shape[2]
    
    del dataSet
    
divI = 1
divPhase = 1
normalizationParFile = Path("Data_Set/normalization_parameters.pickle")
if(normalizationParFile.is_file()):
    normalizationParameters = loadData("normalization_parameters")
    divI = normalizationParameters["divI"]
    divPhase = normalizationParameters["divPhase"]
    