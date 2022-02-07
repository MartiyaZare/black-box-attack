import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def saveData(data, name):
    with open('Data_Set/{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(data, handle)

def loadData(name):
    with open('Data_Set/{}.pickle'.format(name), 'rb') as handle:
        data = pickle.load(handle)
    return data
    
def loadConfig():    
    return loadData("config")

# For original training only -> splits into training, test and validation sets
def getReshapedDataSetThreeSplits(dataSet, modelType):
    dataSetSize = dataSet.shape[0]
    testSetBenignSize = int(dataSetSize/10)
    validationSetSize = int(dataSetSize/10)
    config = loadData("config")
    sequenceLen = config["sequenceLen"]
    dimensionsCount = config["dimensionsCount"]
    
    trainingSetSize = dataSetSize - testSetBenignSize - validationSetSize        
    if(modelType == "fullyConnected" or modelType == "pca"):
        trainingSet = np.reshape(dataSet[0:trainingSetSize,:,:,:], (trainingSetSize,sequenceLen*dimensionsCount), order = 'C')
        testSet = np.reshape(dataSet[trainingSetSize:trainingSetSize+testSetBenignSize,:,:,:], (testSetBenignSize,sequenceLen*dimensionsCount), order = 'C')
        validationSet = np.reshape(dataSet[trainingSetSize+testSetBenignSize:,:,:,:], (validationSetSize,sequenceLen*dimensionsCount), order = 'C')
    elif(modelType == "conv"):        
        trainingSet = np.reshape(dataSet[0:trainingSetSize,:,:,:], (trainingSetSize,sequenceLen*dimensionsCount,1), order = 'C')
        testSet = np.reshape(dataSet[trainingSetSize:trainingSetSize+testSetBenignSize,:,:,:], (testSetBenignSize,sequenceLen*dimensionsCount,1), order = 'C')
        validationSet = np.reshape(dataSet[trainingSetSize+testSetBenignSize:,:,:,:], (validationSetSize,sequenceLen*dimensionsCount,1), order = 'C')
    elif(modelType == "lstm"):
        trainingSet = np.reshape(dataSet[0:trainingSetSize,:,:,:], (trainingSetSize,sequenceLen,dimensionsCount), order = 'C')
        testSet = np.reshape(dataSet[trainingSetSize:trainingSetSize+testSetBenignSize,:,:,:], (testSetBenignSize,sequenceLen,dimensionsCount), order = 'C')
        validationSet = np.reshape(dataSet[trainingSetSize+testSetBenignSize:,:,:,:], (validationSetSize,sequenceLen,dimensionsCount), order = 'C')
        
    return trainingSet, testSet, validationSet

# Reshapes the given dataset without splitting
def getReshapedDataSetNoSplit(dataSet, modelType):
    dataSetSize = dataSet.shape[0]
    config = loadData("config")        
    sequenceLen = config["sequenceLen"]
    dimensionsCount = config["dimensionsCount"]
    testSet = None
    if(modelType == "fullyConnected" or modelType == "pca"):
        testSet = np.reshape(dataSet, (dataSetSize,sequenceLen*dimensionsCount), order = 'C')        
    elif(modelType == "conv"):        
        testSet = np.reshape(dataSet, (dataSetSize,sequenceLen*dimensionsCount,1), order = 'C')
    elif(modelType == "lstm"):      
        testSet = np.reshape(dataSet, (dataSetSize,sequenceLen,dimensionsCount), order = 'C')
    return testSet

# concats advDataSet to dataSet and returns reshaped data without splitting
def getReshapedAdvAddedDataThreeSplits(dataSet, advDataSet, modelType):
    orgTrainingSet, orgTestSet, orgValidationSet = getReshapedDataSetThreeSplits(dataSet, modelType)
    reshapedAdvDataSet = getReshapedDataSetNoSplit(advDataSet, modelType)
    advTrainingSetCount = int(reshapedAdvDataSet.shape[0]/2)    
    advTrainingSet = reshapedAdvDataSet[0:advTrainingSetCount]
    advTestSet = reshapedAdvDataSet[advTrainingSetCount:]
    trainingSet = np.concatenate((orgTrainingSet,advTrainingSet))    
    advLabels = np.zeros(advTrainingSet.shape) # worked 100%
    #advLabels = -advTrainingSet
    labels = np.concatenate((orgTrainingSet,advLabels))   
    return trainingSet, labels, advTestSet, orgValidationSet

def getConfusionMatrix(sortedMSELabels):    
    positiveCount = sum(sortedMSELabels[:,1] == 1)
    TP = sum(sortedMSELabels[0:positiveCount,1] == 1)
    FP = sum(sortedMSELabels[0:positiveCount,1] == 0)
    TN = sum(sortedMSELabels[positiveCount:,1] == 0)
    FN = sum(sortedMSELabels[positiveCount:,1] == 1)
    return TP, FP, TN, FN

def getConfusionMatrixWithThreshold(sortedMSELabels,threshold):
    positiveCount = sum(sortedMSELabels[:,1] == 1)
    TP = sum((sortedMSELabels[:,1] == 1) & (sortedMSELabels[:,0] >= threshold))
    FP = sum((sortedMSELabels[:,1] == 0) & (sortedMSELabels[:,0] >= threshold))
    TN = sum((sortedMSELabels[:,1] == 0) & (sortedMSELabels[:,0] < threshold))
    FN = sum((sortedMSELabels[:,1] == 1) & (sortedMSELabels[:,0] < threshold))
    return TP, FP, TN, FN

def rankedPrecisionAndRecall(mseLabelsList):  
    sortedMSELabels = mseLabelsList[mseLabelsList[:,0].argsort()[::-1]]
    TP, FP, TN, FN = getConfusionMatrix(sortedMSELabels)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    minPositiveMSE = min(sortedMSELabels[sortedMSELabels[:,1] == 1,0]) # 0.010118610983828593
    maxNegativeMSE = float("NaN")
    if(len(sortedMSELabels[sortedMSELabels[:,1] == 0,0])):
        maxNegativeMSE = max(sortedMSELabels[sortedMSELabels[:,1] == 0,0])
    return precision, recall, minPositiveMSE, maxNegativeMSE
    
def rankedPrecisionAndRecallWithThreshold(mseAndLabelsList,threshold):
    sortedMSELabels = mseAndLabelsList[mseAndLabelsList[:,0].argsort()[::-1]]
    TP, FP, TN, FN = getConfusionMatrixWithThreshold(sortedMSELabels,threshold)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall    

# show_curve() is for plotting training and validation losses
def show_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def calculateThreshold(minPos, maxNeg):
    if minPos >= maxNeg:
        return maxNeg + (minPos - maxNeg)/10
    else:
        return minPos
        
def buildAdversarialDataSet(modelType):
    config = loadData("config")    
    adversarialSet = loadData("adversarial_data_set_" + modelType)
    benignDataSet = loadData("normalized_data_set")
    advSetSize = len(adversarialSet)
    advTestSet = adversarialSet[int(advSetSize/2),:]
    _,benignTestSet,_ = getReshapedDataSetThreeSplits(benignDataSet, modelType)
    testSetLabels = np.concatenate((np.ones(len(advTestSet)),np.zeros(len(benignTestSet))))
    testSet = np.concatenate((getReshapedDataSetNoSplit(advTestSet, modelType),benignTestSet))
    return testSet, testSetLabels
    
def plotOneSequenceFromData(dataSet, SampleIndex, FeatureIndex):
    reshapedDataSet = getReshapedDataSetNoSplit(dataSet, "lstm")
    plt.plot(reshapedDataSet[SampleIndex,:,FeatureIndex])
    
# features are mat data keys 'Iprobe1_a', 'Iprobe1_b', 'Iprobe1_c', 'Iprobe2_a', 'Iprobe2_b', 'Iprobe2_c', 'hmod_Iprobe1_a_1_60', 'hpha_Iprobe1_a_1_60', 'hmod_Iprobe2_a_1_60', 'hpha_Iprobe2_a_1_60', 'PhaseAngle_a', 'hmod_Iprobe1_b_1_60', 'hpha_Iprobe1_b_1_60', 'hmod_Iprobe2_b_1_60', 'hpha_Iprobe2_b_1_60', 'PhaseAngle_b', 'PhaseAngle_c', 'SM1_Pe', 'SM2_Pe', 'SM3_Pe', 'SM4_Pe'
# for more info use the following lines:
#   mat = scipy.io.loadmat('Raw_Data/3_Phase_To_Ground/350/1/0001.mat')
#   mat.keys()
def plotRawData(features, faultType = 3, g1Level = 350, g2Level = 350, g3Level = 350, g4Level = 350, randIndex = 1, rang = None, saveFig = 0):
    config = loadData("config")
    rawDataDir = config["rawDataDir"]
    g2 = ((g2Level-350)/2) * 36
    g3 = ((g3Level-350)/2) * 6    
    g4 = ((g3Level-350)/2)
    folderNumber = int(g2 + g3 + g4 + 1)
    mat = scipy.io.loadmat('{}/{}_Phase_To_Ground/{}/{}/00{:02d}.mat'.format(rawDataDir,faultType,g1Level,folderNumber,randIndex))
    if(rang):
        for feature in features:
            plt.plot(mat[feature][rang]) 
    else:
        for feature in features:
            plt.plot(mat[feature]) 
    if(saveFig):
        plt.savefig('raw_data_plot.png')
    