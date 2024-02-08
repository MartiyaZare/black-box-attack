import os
import tqdm
import scipy.io
import random
from MZPackage.Utilities import *
from pathlib import Path


class DataPreprocessor:
    def __init__(self):
        config = loadData("config")
        self.windowLen = config["sequenceLen"]
        self.rawDataDir = config["rawDataDir"]
        self.attackSamplesCount = config["attackSamplesCount"]
        self.dimensionsCount = config["dimensionsCount"]
        self.trainingSetSize = config["trainingSetSize"]
        self.testSetBenignSize = config["testSetBenignSize"]
        # self.testSetSize = config["testSetBenignSize"]

    # finds the max and min of each feature
    # returns two lists
    def rawDataMinMaxFinder(self, featureTitleList, datasetFolderName=None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        maxList = [0] * len(featureTitleList);
        minList = [99999999] * len(featureTitleList)
        maxDict = dict(zip(featureTitleList, maxList))
        minDict = dict(zip(featureTitleList, minList))
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    for feature in featureTitleList:
                                        maxInFile = max(mat[feature]).max()
                                        minInFile = min(mat[feature]).min()
                                        if maxDict[feature] < maxInFile:
                                            maxDict[feature] = maxInFile
                                        if minDict[feature] > minInFile:
                                            minDict[feature] = minInFile
                                p_bar.update(1)
        return maxDict, minDict

    # If fault index not in the data row this function adds Fault_Index field to .mat file
    # basedOn: The field that this function find fault point based on that
    def addFaultTriggerData(self, basedOn, startIndex=4799, endIndex=4950, datasetFolderName=None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    # operation here
                                    initVal = mat[basedOn][0][startIndex]
                                    if "Fault_Index" not in mat.keys():
                                        for i in range(startIndex, endIndex):
                                            diff = abs(mat[basedOn][0][i] - initVal)
                                            if diff > 0.01:
                                                mat[u'Fault_Index'] = i
                                                scipy.io.savemat(filePath, mat)
                                                break
                                p_bar.update(1)

    # If fault index not in the data row this function adds Fault_Index field to .mat file
    # basedOn: The field that this function find fault point based on that
    def secondSystemAddFaultTriggerData(self, basedOn, startIndex=4799, endIndex=4950, datasetFolderName=None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    # operation here
                                    initVal = mat[basedOn][startIndex]
                                    if "Fault_Index" not in mat.keys():
                                        for i in range(startIndex, endIndex):
                                            diff = abs(mat[basedOn][i] - initVal)
                                            if diff > 0.01:
                                                mat[u'Fault_Index'] = i
                                                scipy.io.savemat(filePath, mat)
                                                break
                                p_bar.update(1)

    # Extracts dataset with fault point fixed at center of the sequence
    def extractDataArrayFromRawDataWithFixedFault(self, featureTitleList):
        dataList = []
        for faultType in os.listdir(self.rawDataDir):
            faultTypeDir = '/'.join([self.rawDataDir, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=100) as p_bar:
                            total = len(os.listdir(generationLevelDir))
                            processed = 0
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    # operation here
                                    toAddSubSequence = []
                                    faultIndex = mat["Fault_Index"][0][0]
                                    startIndex = int(faultIndex - self.windowLen / 2)
                                    endIndex = int(faultIndex + self.windowLen / 2)
                                    for feature in featureTitleList:
                                        toAddSubSequence.append(list(mat[feature][startIndex:endIndex]))
                                    dataList.append(toAddSubSequence)
                                processed += 1
                                p_bar.update(processed / total)
        return np.array(dataList)



    def normalizeDataSetAndSaveParams(self, dataArray, phase=0):
        dataSize = dataArray.shape[0]
        flatArrSize = dataSize * self.windowLen
        iProbe1A = dataArray[:, :, 0].reshape([flatArrSize])
        iProbe1B = dataArray[:, :, 1].reshape([flatArrSize])
        iProbe1C = dataArray[:, :, 2].reshape([flatArrSize])
        iProbe2A = dataArray[:, :, 3].reshape([flatArrSize])
        iProbe2B = dataArray[:, :, 4].reshape([flatArrSize])
        iProbe2C = dataArray[:, :, 5].reshape([flatArrSize])
        maxI = max([max(iProbe1A), max(iProbe1B), max(iProbe1C), max(iProbe2A), max(iProbe2B), max(iProbe2C)])
        minI = min([min(iProbe1A), min(iProbe1B), min(iProbe1C), min(iProbe2A), min(iProbe2B), min(iProbe2C)])
        divI = max([abs(maxI), abs(minI)])
        divPhase = 180
        normalizationParameters = {"divI": divI, "divPhase": divPhase}
        with open('Data_Set/normalization_parameters.pickle', 'wb') as handle:
            pickle.dump(normalizationParameters, handle)
        print("divI is %d" % divI)
        print("divPhase is %d" % divPhase)
        dataArray[:, :, 0:6] = dataArray[:, :, 0:6] / divI
        if phase == 1:
            dataArray[:, :, 6:9] = dataArray[:, :, 6:9] / divPhase
        return dataArray


    def normalizeDataSetUsingSavedParams(self, dataArray, phase=0):
        normalizationParameters = loadData("normalization_parameters")
        divI = normalizationParameters["divI"]
        divPhase = normalizationParameters["divPhase"]
        dataArray[:, :, 0:6] = dataArray[:, :, 0:6] / divI
        if phase == 1:
            dataArray[:, :, 6:9] = dataArray[:, :, 6:9] / divPhase
        return dataArray

    def secondSystemNormalizeDataSetUsingSavedParams(self, dataArray, phase=0):
        normalizationParameters = loadData("normalization_parameters")
        divI = normalizationParameters["divI"]
        divPhase = normalizationParameters["divPhase"]
        dataArray[:, :, 0:6] = dataArray[:, :, 0:6] / divI
        if phase == 1:
            dataArray[:, :, 6:9] = dataArray[:, :, 6:9] / divPhase
        return dataArray

    def getNoFaultData(self, featureTitleList, datasetFolderName=None, windowLenParam = None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        windowLen = self.windowLen
        if windowLenParam:
            windowLen = windowLenParam
        dataList = []
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if (os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if (os.path.isdir(generationLevelDir)):
                        ttl = len(os.listdir(generationLevelDir))
                        with tqdm.tqdm(total=ttl) as p_bar:
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    toAddSubSequence = []
                                    faultIndex = mat["Fault_Index"][0][0]
                                    randInt = random.randint(1, 101)
                                    startIndex = 500 + randInt
                                    endIndex = 500 + windowLen + randInt
                                    for feature in featureTitleList:
                                        toAppendArr = mat[feature][0][startIndex:endIndex]
                                        toAddSubSequence.append(toAppendArr)
                                    dataList.append(toAddSubSequence)
                                p_bar.update(1)
        return np.swapaxes(np.array(dataList), 1, 2)

    def getShiftAttackData(self, featureTitleList, featuresToShift, count, shiftRange=[20, 40], datasetFolderName=None, windowlenParam=None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        windowLen = self.windowLen
        if windowlenParam:
            windowLen = windowlenParam
        dataList = []
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if (os.path.isdir(faultTypeDir)):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if (os.path.isdir(generationLevelDir)):
                        ttl = len(os.listdir(generationLevelDir))
                        with tqdm.tqdm(total=ttl) as p_bar:
                            processed = 0;
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    toAddSubSequence = []
                                    randInt = random.randint(1, 101)
                                    startIndex = 500 + randInt
                                    endIndex = 500 + windowLen + randInt
                                    shiftBy = random.randint(shiftRange[0], shiftRange[1])
                                    for feature in featureTitleList:
                                        toAppendArr = mat[feature][0][startIndex:endIndex]
                                        if (feature in featuresToShift):
                                            shiftedHalfStartIndex = int(startIndex + windowLen / 2 - shiftBy)
                                            shiftedHalfEndIndex = int(endIndex - shiftBy)
                                            attackStartIndex = int(windowLen / 2)
                                            toAppendArr[attackStartIndex:] = mat[feature][0][
                                                                             shiftedHalfStartIndex:shiftedHalfEndIndex]
                                        toAddSubSequence.append(toAppendArr)
                                    dataList.append(toAddSubSequence)
                                p_bar.update(1)
        smapledList = random.sample(dataList, count)
        return np.swapaxes(np.array(smapledList), 1, 2)

    def getRandomFDIAttackData(self, featuresIndexesToAttack, count, noFaultDataSet):
        benignData = loadData(noFaultDataSet)
        sampledBenignDataList = random.sample(list(benignData), count)
        sampledBenignDataArr = np.array(sampledBenignDataList)
        ttl = len(sampledBenignDataArr)
        with tqdm.tqdm(total=ttl) as p_bar:
            processed = 0;
            for dataSeq in sampledBenignDataArr:
                for featureIndex in featuresIndexesToAttack:
                    halfWindowLen = self.windowLen // 2
                    randArr = (np.random.rand(count, halfWindowLen) - 0.5) * 2
                    sampledBenignDataArr[:, halfWindowLen:, featureIndex] = randArr
                p_bar.update(1)  # One is processed

        return sampledBenignDataArr

    def getReplayAttackData(self, count, noFaultDataSet, faultDataSet, windowLenParam=None):
        windowLen = self.windowLen
        if windowLenParam:
            windowLen = windowLenParam
        benignData = loadData(noFaultDataSet)
        faultData = loadData(faultDataSet)
        sampledBenignDataList = random.sample(list(benignData), count)
        sampledBenignDataArr = np.array(sampledBenignDataList)
        sampledFaultDataList = random.sample(list(faultData), count)
        sampledFaultDataArr = np.array(sampledFaultDataList)
        # Add attack
        # tmp
        coef = 1.4
        halfWindowLen = windowLen // 2
        sampledBenignDataArr[:, halfWindowLen:, 0:3] = coef * sampledFaultDataArr[:, halfWindowLen:, 0:3]
        return sampledBenignDataArr

    def getAttackData(self, dataSetfolderParam=None):
        dataSetfolder = ""
        faultDataSetAddr = "normalized_data_set"
        fdiDataSetAddr = "normalized_fdi_attack_data_set"
        timeSyncDataSetAddr = "normalized_time_sync_attack_data_set"
        replayDataSetAddr = "normalized_replay_attack_data_set"
        if dataSetfolderParam:
            dataSetfolder = dataSetfolderParam
            faultDataSetAddr = "/".join([dataSetfolder, "normalized_data_set"])
            fdiDataSetAddr = "/".join([dataSetfolder, "normalized_fdi_attack_data_set"])
            timeSyncDataSetAddr = "/".join([dataSetfolder, "normalized_time_sync_attack_data_set"])
            replayDataSetAddr = "/".join([dataSetfolder, "normalized_replay_attack_data_set"])
        print(faultDataSetAddr)
        normalizedDataSet = loadData(faultDataSetAddr)
        normalizedFDIAttackDataSet = loadData(fdiDataSetAddr)
        normalizedTimeSyncAttackDataSet = loadData(timeSyncDataSetAddr)
        normalizedReplayAttackDataSet = loadData(replayDataSetAddr)
        # tmp
        ####attackDataSetsTuple = (normalizedAngleShiftAttackDataSet, normalizedFDIAttackDataSet, normalizedTimeSyncAttackDataSet, normalizedReplayAttackDataSet)
        attackDataSetsTuple = (normalizedFDIAttackDataSet, normalizedTimeSyncAttackDataSet, normalizedReplayAttackDataSet)

        # for now we mix all three attack data sets into one
        attackDataSet = np.concatenate(attackDataSetsTuple, axis=0)
        sampledAttackDataSet = np.array(random.sample(list(attackDataSet), self.attackSamplesCount))
        # sampledAttackDataSet = attackDataSet
        testSetBenign = np.reshape(
            normalizedDataSet[self.trainingSetSize:self.trainingSetSize + self.testSetBenignSize, :, :],
            (self.testSetBenignSize, self.windowLen, self.dimensionsCount), order='C')
        testSetAttack = np.reshape(sampledAttackDataSet,
                                   (self.attackSamplesCount, self.windowLen, self.dimensionsCount), order='C')
        testSet = np.concatenate((testSetBenign, testSetAttack))
        testSetSize = self.testSetBenignSize + self.attackSamplesCount
        testSetLabels = np.zeros(testSetSize)
        testSetLabels[self.testSetBenignSize:] = np.ones(self.attackSamplesCount)
        saveData(testSet, "/".join([dataSetfolder, "sampled_normal_and_attack_test_set"]))
        saveData(testSetLabels, "/".join([dataSetfolder, "sampled_normal_and_attack_test_set_labels"]))
        return testSet, testSetLabels

    def getOldAttackData(self):
        return loadData("sampled_normal_and_attack_test_set"), loadData("sampled_normal_and_attack_test_set_labels")

    def datToMatBatch(self, srcFolder=None, dstFolder=None, linesToSkipInConf=2):
        if srcFolder is None:
            srcFolder = self.rawDataDir
        if dstFolder is None:
            dstFolder = "/".join([srcFolder, "matFiles"])
        # find coeffs and column titles
        print(srcFolder)
        if os.path.isdir(srcFolder):
            titles = []
            ACoeffs = []
            BBiases = []
            dataSequence = []
            for fileName in os.listdir(srcFolder):
                if fileName.endswith('.cfg'):
                    titles = []
                    ACoeffs = []
                    BBiases = []
                    file = open("{}/{}".format(srcFolder, fileName))
                    for lineIndex, line in enumerate(file):
                        if lineIndex >= linesToSkipInConf:  # to skip the first 2 lines
                            result = line.split(',')
                            if len(result) > 1:  # to skip unrelated lines in the end of the config file
                                titles.append(result[1].replace('.', '_'))
                                ACoeffs.append(result[5])
                                BBiases.append(result[6])
                            else:
                                break
                    file.close()
                elif fileName.endswith('.dat'):
                    dataSequence = []
                    file = open("{}/{}".format(srcFolder, fileName))
                    for lineIndex, line in enumerate(file):
                        result = line.split(',')
                        for index, entry in enumerate(result[2:]):
                            originalValue = float(entry) * float(ACoeffs[index]) + float(BBiases[index])
                            result[index] = originalValue
                        dataSequence.append(result)
                    file.close()
                    dataSequenceArr = np.asarray(dataSequence, dtype=float)
                    matDict = {}
                    for index, title in enumerate(titles):
                        matDict[title] = dataSequenceArr[:, index]
                    matFileName = fileName.split(".")[0]
                    if not os.path.isdir(dstFolder):
                        Path(dstFolder).mkdir(parents=True, exist_ok=True)
                    scipy.io.savemat('/'.join([dstFolder, matFileName + '.mat']), matDict)

    # Extracts dataset from windows ending right at fault point (no-fault right before fault)
    def extractDataArrayNormalAndFault(self, featureTitleList, winLen=200, datasetFolderName=None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        dataListBefore = []
        dataListAfter = []
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:
                            processed = 0
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    # operation here
                                    toAddSubSequenceNormal = []
                                    toAddSubSequenceFault = []
                                    faultIndex = mat["Fault_Index"][0][0]
                                    startIndexNormal = int(faultIndex - winLen)
                                    endIndexNormal = int(faultIndex)
                                    endIndexFault = int(faultIndex) + winLen
                                    for feature in featureTitleList:
                                        toAddSubSequenceNormal.append(list(mat[feature][0][startIndexNormal:endIndexNormal]))
                                        toAddSubSequenceFault.append(list(mat[feature][0][endIndexNormal:endIndexFault]))
                                    dataListBefore.append(toAddSubSequenceNormal)
                                    dataListAfter.append(toAddSubSequenceFault)
                                p_bar.update(1)
        return np.array(dataListBefore), np.array(dataListAfter)

    # Extracts dataset from windows ending right at fault point (no-fault right before fault)
    def secondSystemExtractDataArrayNormalAndFault(self, featureTitleList, winLen=200, datasetFolderName=None):
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        dataListBefore = []
        dataListAfter = []
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:
                            processed = 0
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    # operation here
                                    toAddSubSequenceNormal = []
                                    toAddSubSequenceFault = []
                                    faultIndex = mat["Fault_Index"][0]
                                    startIndexNormal = int(faultIndex - winLen)
                                    endIndexNormal = int(faultIndex)
                                    endIndexFault = int(faultIndex) + winLen
                                    for feature in featureTitleList:
                                        toAddSubSequenceNormal.append(
                                            list(mat[feature][startIndexNormal:endIndexNormal]))
                                        toAddSubSequenceFault.append(
                                            list(mat[feature][endIndexNormal:endIndexFault]))
                                    dataListBefore.append(toAddSubSequenceNormal)
                                    dataListAfter.append(toAddSubSequenceFault)
                                p_bar.update(1)
        return np.array(dataListBefore), np.array(dataListAfter)

    # Extracts dataset with fault point randomized shifts
    def extractDataArrayFromRawDataMovingWindow(self, featureTitleList, datasetFolderName=None, windowLen=None):
        if windowLen is None:
            windowLen = self.windowLen
        datasetLocation = self.rawDataDir
        if datasetFolderName:
            datasetLocation = '/'.join([datasetLocation, datasetFolderName])
        dataList = []
        for faultType in os.listdir(datasetLocation):
            faultTypeDir = '/'.join([datasetLocation, faultType])
            if os.path.isdir(faultTypeDir):
                for generationLevel in os.listdir(faultTypeDir):
                    generationLevelDir = '/'.join([faultTypeDir, generationLevel])
                    if os.path.isdir(generationLevelDir):
                        with tqdm.tqdm(total=len(os.listdir(generationLevelDir))) as p_bar:
                            processed = 0
                            for dirName in os.listdir(generationLevelDir):
                                dataDir = '/'.join([generationLevelDir, dirName])
                                for fileName in os.listdir(dataDir):
                                    filePath = '/'.join([dataDir, fileName])
                                    mat = scipy.io.loadmat(filePath)
                                    # operation here
                                    toAddSubSequence = []
                                    faultIndex = mat["Fault_Index"][0][0]
                                    randShift = random.randrange(int(-windowLen / 4), int(windowLen / 4))
                                    startIndex = int(faultIndex - windowLen / 2 - randShift)
                                    endIndex = int(faultIndex + windowLen / 2 - randShift)
                                    for feature in featureTitleList:
                                        toAddSubSequence.append(list(mat[feature][0][startIndex:endIndex]))
                                    dataList.append(toAddSubSequence)

                                p_bar.update(1)
        return np.array(dataList)
