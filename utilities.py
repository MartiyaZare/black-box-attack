import pickle
import numpy as np

def saveData(data, name):
    with open('Data_Set/{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(data, handle)

def loadData(name):
    with open('Data_Set/{}.pickle'.format(name), 'rb') as handle:
        data = pickle.load(handle)
    return data

# For original training only -> splits into training, test and validation sets
def getReshapedDataSet(dataSet, modelType):
    dataSetSize = dataSet.shape[0]
    testSetBenignSize = int(dataSetSize/10)
    validationSetSize = int(dataSetSize/10)
    config = loadData("config")
    sequenceLen = config["sequenceLen"]
    dimensionsCount = config["dimensionsCount"]
    
    trainingSetSize = dataSetSize - testSetBenignSize - validationSetSize        
    if(modelType == "fullyConnected" or modelType == "PCA"):
        trainingSet = np.reshape(dataSet[0:trainingSetSize,:,:,:], (trainingSetSize,sequenceLen*dimensionsCount), order = 'C')
        testSet = np.reshape(dataSet[trainingSetSize:trainingSetSize+testSetBenignSize,:,:,:], (testSetBenignSize,sequenceLen*dimensionsCount), order = 'C')
        validationSet = np.reshape(dataSet[trainingSetSize+testSetBenignSize:,:,:,:], (validationSetSize,sequenceLen*dimensionsCount), order = 'C')
    elif(modelType == "1DConv"):        
        trainingSet = np.reshape(dataSet[0:trainingSetSize,:,:,:], (trainingSetSize,sequenceLen*dimensionsCount,1), order = 'C')
        testSet = np.reshape(dataSet[trainingSetSize:trainingSetSize+testSetBenignSize,:,:,:], (testSetBenignSize,sequenceLen*dimensionsCount,1), order = 'C')
        validationSet = np.reshape(dataSet[trainingSetSize+testSetBenignSize:,:,:,:], (validationSetSize,sequenceLen*dimensionsCount,1), order = 'C')
    elif(modelType == "LSTM"):
        trainingSet = np.reshape(dataSet[0:trainingSetSize,:,:,:], (trainingSetSize,sequenceLen,dimensionsCount), order = 'C')
        testSet = np.reshape(dataSet[trainingSetSize:trainingSetSize+testSetBenignSize,:,:,:], (testSetBenignSize,sequenceLen,dimensionsCount), order = 'C')
        validationSet = np.reshape(dataSet[trainingSetSize+testSetBenignSize:,:,:,:], (validationSetSize,sequenceLen,dimensionsCount), order = 'C')
        
    return trainingSet, testSet, validationSet

# Reshapes the given dataset without splitting
def getReshapedTestSet(dataSet, modelType):
    dataSetSize = dataSet.shape[0]
    config = loadData("config")        
    sequenceLen = config["sequenceLen"]
    dimensionsCount = config["dimensionsCount"]
    
    if(modelType == "fullyConnected" or modelType == "PCA"):
        testSet = np.reshape(dataSet, (dataSetSize,sequenceLen*dimensionsCount), order = 'C')        
    elif(modelType == "1DConv"):        
        testSet = np.reshape(dataSet, (dataSetSize,sequenceLen*dimensionsCount,1), order = 'C')
    elif(modelType == "LSTM"):      
        testSet = np.reshape(dataSet, (dataSetSize,sequenceLen,dimensionsCount), order = 'C')
    return testSet

# Reshapes the given dataset without splitting
def getReshapedAdversarialDataSet(dataSet, advDataSet, modelType):
    orgTrainingSet, orgTestSet, orgValidationSet = getReshapedDataSet(dataSet, modelType)
    reshapedAdvDataSet = getReshapedTestSet(advDataSet, modelType)
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
