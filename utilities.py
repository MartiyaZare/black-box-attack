import pickle

def saveData(data, name):
    with open('Data_Set/{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(data, handle)

def loadData(name):
    with open('Data_Set/{}.pickle'.format(name), 'rb') as handle:
        data = pickle.load(handle)
    return data

def getReshapedDataSet(dataset, modelType):
    dataSetSize = dataSet.shape[0]
    testSetBenignSize = int(dataSetSize/10)
    validationSetSize = int(dataSetSize/10)
    trainingSetSize = dataSetSize - testSetBenignSize - validationSetSize
    
    config = {"dataSetSize":dataSetSize,
              "testSetBenignSize":testSetBenignSize,
              "validationSetSize":validationSetSize,
              "trainingSetSize":trainingSetSize,
              "sequenceLen":sequenceLen,
              "dimensionsCount":dimensionsCount}
    saveData(config, "config")    
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

