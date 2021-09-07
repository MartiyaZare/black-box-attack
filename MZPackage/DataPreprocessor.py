import os
import tqdm
import scipy.io
import random
from MZPackage.Utilities import *


class DataPreprocessor:
    def __init__(self):
        config = loadData("config")
        self.windowLen = config["sequenceLen"]
        self.rawDataDir = config["rawDataDir"]
        self.attackSamplesCount = config["attackSamplesCount"]
        self.dimensionsCount = config["dimensionsCount"]
        self.trainingSetSize = config["trainingSetSize"]
        self.testSetBenignSize = config["testSetBenignSize"]
        #self.testSetSize = config["testSetBenignSize"]

    # finds the max and min of each feature
    # returns two lists
    def rawDataMinMaxFinder(self, featureTitleList):
        maxList = [0] * len(featureTitleList);
        minList = [99999999] * len(featureTitleList)
        maxDict = dict(zip(featureTitleList, maxList))
        minDict = dict(zip(featureTitleList, minList))        
        for faultType in os.listdir(self.rawDataDir):
            faultTypeDir = '/'.join([self.rawDataDir,faultType])
            if(os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir,generationLevel]) 
                    if(os.path.isdir(generationLevelDir)):                     
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:                            
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir,dirName])                                                    
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir,fileName]) 
                                    mat = scipy.io.loadmat(filePath)
                                    for feature in featureTitleList:
                                        maxInFile = max(mat[feature])[0]
                                        minInFile = min(mat[feature])[0]
                                        if(maxDict[feature] < maxInFile):
                                            maxDict[feature] = maxInFile
                                        if(minDict[feature] > minInFile):
                                            minDict[feature] = minInFile                            
                                p_bar.update(1)                                
        return maxDict,minDict
        

    # If fault index not in the data row this function adds Fault_Index field to .mat file
    # basedOn: The field that this function find fault point based on that
    def addFaultTriggerData(self, basedOn):    
        for faultType in os.listdir(self.rawDataDir):
            faultTypeDir = '/'.join([self.rawDataDir,faultType])
            if(os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir,generationLevel]) 
                    if(os.path.isdir(generationLevelDir)):   
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir)) ) as p_bar:                        
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir,dirName])                                                    
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir,fileName]) 
                                    mat = scipy.io.loadmat(filePath)
                                    #operation here
                                    initVal = mat[basedOn][20000]
                                    if("Fault_Index" not in mat.keys()):                                    
                                        for i in range(20000,20405):
                                            diff = abs(mat[basedOn][i] - initVal)
                                            if(diff > 0.01):
                                                mat[u'Fault_Index'] = i
                                                scipy.io.savemat(filePath, mat)                                            
                                                break                                                          
                                p_bar.update(1) 

    # Extracts dataset with fault point fixed at center of the sequence
    def extractDataArrayFromRawDataWithFixedFault(self, featureTitleList):
        dataList = []
        for faultType in os.listdir(rawDataDir):
            faultTypeDir = '/'.join([rawDataDir,faultType])
            if(os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir,generationLevel]) 
                    if(os.path.isdir(generationLevelDir)):   
                        with tqdm.tqdm(total=100) as p_bar:
                            total = len(os.listdir(generationLevelDir))     
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir,dirName])                                                    
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir,fileName]) 
                                    mat = scipy.io.loadmat(filePath)
                                    #operation here
                                    toAddSubSequence = []                                
                                    faultIndex = mat["Fault_Index"][0][0]                                
                                    startIndex = int(faultIndex - self.windowLen/2)
                                    endIndex = int(faultIndex + self.windowLen/2)                                
                                    for feature in featureTitleList:                                                                        
                                        toAddSubSequence.append(list(mat[feature][startIndex:endIndex]))  
                                    dataList.append(toAddSubSequence)
                                processed += 1
                                p_bar.update(processed/total) 
        return np.array(dataList)      

    # Extracts dataset with fault point randomized shifts
    def extractDataArrayFromRawDataMovingWindow(self, featureTitleList):
        dataList = []
        for faultType in os.listdir(self.rawDataDir):
            faultTypeDir = '/'.join([self.rawDataDir,faultType])
            if(os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir,generationLevel]) 
                    if(os.path.isdir(generationLevelDir)):   
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir)) ) as p_bar:                        
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir,dirName])                                                    
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir,fileName]) 
                                    mat = scipy.io.loadmat(filePath)
                                    #operation here
                                    toAddSubSequence = []                                
                                    faultIndex = mat["Fault_Index"][0][0]  
                                    randShift = random.randrange(int(-self.windowLen/4), int(self.windowLen/4))
                                    startIndex = int(faultIndex - self.windowLen/2 - randShift)
                                    endIndex = int(faultIndex + self.windowLen/2 - randShift)                                
                                    for feature in featureTitleList:                                                                        
                                        toAddSubSequence.append(list(mat[feature][startIndex:endIndex]))  
                                    dataList.append(toAddSubSequence)                            
                                p_bar.update(1) 
        return np.array(dataList) 

    def normalizeDataSetAndSaveParams(self, dataArray):
        dataSize = dataArray.shape[0]
        flatArrSize = dataSize*self.windowLen
        iProbe1A = dataArray[:,:,0,0].reshape([flatArrSize])
        iProbe1B = dataArray[:,:,1,0].reshape([flatArrSize])
        iProbe1C = dataArray[:,:,2,0].reshape([flatArrSize])
        iProbe2A = dataArray[:,:,3,0].reshape([flatArrSize])
        iProbe2B = dataArray[:,:,4,0].reshape([flatArrSize])
        iProbe2C = dataArray[:,:,5,0].reshape([flatArrSize])
        maxI = max([max(iProbe1A),max(iProbe1B),max(iProbe1C),max(iProbe2A),max(iProbe2B),max(iProbe2C)])
        minI = min([min(iProbe1A),min(iProbe1B),min(iProbe1C),min(iProbe2A),min(iProbe2B),min(iProbe2C)])
        divI = max([abs(maxI),abs(minI)])
        divPhase = 180
        normalizationParameters = {"divI": divI, "divPhase": divPhase}
        with open('Data_Set/normalization_parameters.pickle', 'wb') as handle:
            pickle.dump(normalizationParameters, handle)
        print("divI is %d"%divI)
        print("divPhase is %d"%divPhase)
        dataArray[:,:,0:6,0] = dataArray[:,:,0:6,0]/divI
        dataArray[:,:,6:9,0] = dataArray[:,:,6:9,0]/divPhase        
        return dataArray
        
    def normalizeDataSetUsingSavedParams(self, dataArray):    
        normalizationParameters = loadData("normalization_parameters")    
        divI = normalizationParameters["divI"]
        divPhase = normalizationParameters["divPhase"]
        dataArray[:,:,0:6,0] = dataArray[:,:,0:6,0]/divI
        dataArray[:,:,6:9,0] = dataArray[:,:,6:9,0]/divPhase    
        return dataArray
        
    def getBenignData(self, featureTitleList):
        dataList = []
        for faultType in os.listdir(self.rawDataDir):
            faultTypeDir = '/'.join([self.rawDataDir,faultType])
            if(os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir,generationLevel]) 
                    if(os.path.isdir(generationLevelDir)):
                        ttl = len(os.listdir(generationLevelDir))
                        with tqdm.tqdm(total=ttl) as p_bar:                             
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir,dirName])                                                    
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir,fileName]) 
                                    mat = scipy.io.loadmat(filePath)                               
                                    toAddSubSequence = []                                
                                    faultIndex = mat["Fault_Index"][0][0]                                                                  
                                    randInt = random.randint(1,101)
                                    startIndex = 500 + randInt
                                    endIndex = 700 + randInt                                    
                                    for feature in featureTitleList:  
                                        toAppendArr = mat[feature][startIndex:endIndex]                                                                        
                                        toAddSubSequence.append(toAppendArr) 
                                    dataList.append(toAddSubSequence)                            
                                p_bar.update(1)    
        return np.swapaxes(np.array(dataList), 1, 2)
        
    def getShiftedAngleData(self, featureTitleList, featuresToShift, count, shiftRange = 40):
        dataList = []
        for faultType in os.listdir(self.rawDataDir):
            faultTypeDir = '/'.join([self.rawDataDir,faultType])
            if(os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir,generationLevel]) 
                    if(os.path.isdir(generationLevelDir)):      
                        ttl = len(os.listdir(generationLevelDir)) 
                        with tqdm.tqdm(total=ttl) as p_bar:                             
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir,dirName])                                                    
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir,fileName]) 
                                    mat = scipy.io.loadmat(filePath)                               
                                    toAddSubSequence = []                                                                                                                                  
                                    randInt = random.randint(1,101)
                                    startIndex = 500 + randInt
                                    endIndex = 700 + randInt    
                                    shiftBy = random.randint(20,shiftRange)
                                    for feature in featureTitleList:  
                                        toAppendArr = mat[feature][startIndex:endIndex]                                    
                                        if(feature in featuresToShift):
                                            shiftedHalfStartIndex = int(startIndex+self.windowLen/2-shiftBy)
                                            shiftedHalfEndIndex = int(endIndex-shiftBy)
                                            attackStartIndex = int(self.windowLen/2)
                                            toAppendArr[attackStartIndex:] = mat[feature][shiftedHalfStartIndex:shiftedHalfEndIndex]
                                        toAddSubSequence.append(toAppendArr) 
                                    dataList.append(toAddSubSequence)                                                        
                                p_bar.update(1) 
        smapledList = random.sample(dataList, count)
        return np.swapaxes(np.array(smapledList), 1, 2)
        
    def getRandomFDIAttackData(self, featuresIndexesToAttack, count):
        benignData = loadData("normalized_benign_data_set")
        sampledBenignDataList = random.sample(list(benignData), count)
        sampledBenignDataArr = np.array(sampledBenignDataList)
        ttl = len(sampledBenignDataArr)
        with tqdm.tqdm(total=ttl) as p_bar:              
            processed = 0;  
            for dataSeq in sampledBenignDataArr:                                                                                                        
                for featureIndex in featuresIndexesToAttack:  
                    randArr = np.random.rand(count,self.windowLen)                
                    sampledBenignDataArr[:,:,featureIndex,0] = randArr                                                                 
                p_bar.update(1) # One is processed
                
        return sampledBenignDataArr
        
    def getAttackData(self):
        normalizedDataSet = loadData("normalized_data_set")
        normalizedAngleShiftAttackDataSet = loadData("normalized_angle_shift_attack_data_set")
        normalizedFDIAttackDataSet = loadData("normalized_fdi_attack_data_set")
        normalizedTimeSyncAttackDataSet = loadData("normalized_time_sync_attack_data_set")

        # for now we mix all three attack data sets into one
        attackDataSet = np.concatenate((normalizedAngleShiftAttackDataSet, normalizedFDIAttackDataSet, normalizedTimeSyncAttackDataSet), axis=0)
        sampledAttackDataSet = np.array(random.sample(list(attackDataSet), self.attackSamplesCount))
        #sampledAttackDataSet = attackDataSet
        testSetBenign = np.reshape(normalizedDataSet[self.trainingSetSize:self.trainingSetSize+self.testSetBenignSize,:,:,:], (self.testSetBenignSize,self.windowLen,self.dimensionsCount), order = 'C')
        testSetAttack = np.reshape(sampledAttackDataSet, (self.attackSamplesCount,self.windowLen,self.dimensionsCount), order = 'C')
        testSet = np.concatenate((testSetBenign,testSetAttack))
        testSetSize = self.testSetBenignSize + self.attackSamplesCount
        testSetLabels = np.zeros(testSetSize)
        testSetLabels[self.testSetBenignSize:] = np.ones(self.attackSamplesCount)
        return testSet, testSetLabels