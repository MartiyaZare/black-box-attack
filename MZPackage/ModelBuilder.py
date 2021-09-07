from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM, Flatten, Reshape, ZeroPadding1D, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.keras import backend as K
from MZPackage.Utilities import *

class ModelBuilder:
    def __init__(self):
        config = loadConfig()        
        self.sequenceLen = config["sequenceLen"]
        self.dimensionsCount = config["dimensionsCount"]
        self.attackSamplesCount = config["attackSamplesCount"]        
        self.testSetBenignSize = config["testSetBenignSize"]


    def trainPca(self, codeSize, trainingSet, trainingSetLabels, validationSet, validationSetLabels, printSummary = 1, vrbs = 1, return_best = 1, learningRate = 0.001, epochsParam = 1000, batchSize = 5000):
        inputSize = self.sequenceLen * self.dimensionsCount
        model = Sequential()
        model.add(Dense(inputSize, input_shape=(inputSize,), activation='linear'))
        model.add(Dense(int(codeSize), activation="linear"))
        model.add(Dense(inputSize, activation="linear"))
        if(printSummary == True):
                model.summary()
        adamOptimizer = Adam(learning_rate=learningRate) # , beta_1=0.9, beta_2=0.999, amsgrad=False        
        model.compile(optimizer=adamOptimizer,
            loss='mean_squared_error',
        )
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
        callbacksArray = [es]
        if(return_best):
            mc = ModelCheckpoint('best_pca.h5', monitor='val_loss', mode='min')
            callbacksArray = [es, mc]        
        history=model.fit(trainingSet, trainingSetLabels,
                                batch_size=batchSize,
                                shuffle=True,
                                epochs=epochsParam,                             
                                validation_data=(validationSet, validationSetLabels),
                                callbacks=callbacksArray,
                                verbose = vrbs,
                            )    
        if(return_best):
            best_model = load_model('best_pca.h5')        
        returnModel = model
        if(return_best):
            returnModel = best_model
        return [returnModel,min(history.history['val_loss']),len(history.history['val_loss']),history]
        
        
    def trainFullyConnected(self, numOfHiddenLayersInEncoder, codeLayerSize, trainingSet, trainingSetLabels, validationSet, validationSetLabels, printSummary = 1, vrbs = 1, return_best = 1, learningRate = 0.001, epochsParam = 1000, batchSize = 5000):
        inputSize = self.sequenceLen * self.dimensionsCount
        layerSizeDifference = (inputSize - codeLayerSize) // (numOfHiddenLayersInEncoder + 1)
        model = Sequential()        
        model.add(Dense(inputSize, input_shape=(inputSize,), activation='relu'))
        for i in range(1,numOfHiddenLayersInEncoder + 1):
            model.add(Dense(int(inputSize - (layerSizeDifference * i)), activation="relu"))        
        model.add(Dense(int(codeLayerSize), activation="relu"))        
        for j in range(1,numOfHiddenLayersInEncoder + 1):    
            model.add(Dense(int(inputSize - (layerSizeDifference * (numOfHiddenLayersInEncoder - j + 1))), activation="relu"))        
        model.add(Dense(inputSize, activation="linear"))
        if(printSummary == True):
                model.summary()
        adamOptimizer = Adam(learning_rate=learningRate) # , beta_1=0.9, beta_2=0.999, amsgrad=False                  
        model.compile(optimizer=adamOptimizer,
            loss='mean_squared_error',
        )    
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
        callbacksArray = [es]
        if(return_best):
            mc = ModelCheckpoint('best_Autoencoder.h5', monitor='val_loss', mode='min')
            callbacksArray = [es, mc]    
        history=model.fit(trainingSet, trainingSetLabels,
                                batch_size=batchSize,
                                shuffle=True,
                                epochs=epochsParam,                             
                                validation_data=(validationSet, validationSetLabels),
                                callbacks=callbacksArray,
                                verbose = vrbs,
                            )    
        if(return_best):
            best_model = load_model('best_Autoencoder.h5')        
        returnModel = model
        if(return_best):
            returnModel = best_model
        return [returnModel,min(history.history['val_loss']),len(history.history['val_loss']),history]
        
    def trainConv(self, numOfHiddenLayersInEncoder, trainingSet, trainingSetLabels, validationSet, validationSetLabels, filtersCountInFirstLayer = 32, printSummary = 1, vrbs = 1, return_best = 1, filterSize = 3, learningRate = 0.001, epochsParam = 10000, batchSize = 1000):
        numOfHiddenLayersInEncoder = numOfHiddenLayersInEncoder + 1
        model = Sequential()    
        poolingSize = 2
        numOfFiltersInEncoder = [filtersCountInFirstLayer]
        paddingSize = (filterSize // 2)
        inputLen = (self.sequenceLen * self.dimensionsCount)
        paddedInputLength = inputLen + (paddingSize*2)
        encoderLayersFilterSizes = [paddedInputLength]
        model.add(ZeroPadding1D(paddingSize,input_shape=(self.sequenceLen * self.dimensionsCount,1)))
        model.add(Conv1D(int(filtersCountInFirstLayer), filterSize, activation='relu', padding = 'same'))
        for i in range(1,numOfHiddenLayersInEncoder):
            model.add(MaxPooling1D(poolingSize, padding='same'))
            model.add(Conv1D(int(filtersCountInFirstLayer*np.power(2,i)), filterSize, activation='relu', padding = 'same'))          
            encoderLayersFilterSizes.append(int(np.ceil(paddedInputLength/np.power(2,i))))
            numOfFiltersInEncoder.append(int(filtersCountInFirstLayer*np.power(2,i)))
        model.add(Flatten())
        model.add(Reshape((encoderLayersFilterSizes[-1], numOfFiltersInEncoder[-1])))
        for j in range(1,numOfHiddenLayersInEncoder):    
            model.add(Conv1D(numOfFiltersInEncoder[-j], filterSize, activation='relu', padding = 'same'))
            model.add(UpSampling1D(poolingSize))
        model.add(Conv1D(numOfFiltersInEncoder[0], filterSize, activation='relu', padding='same'))
        model.add(Conv1D(1, filterSize, activation='linear', padding='same'))        
        toCrop = (model.layers[-1].output_shape[1] - inputLen) // 2        
        model.add(Cropping1D(toCrop))
        model.summary()
        adamOptimizer = Adam(learning_rate=learningRate) # , beta_1=0.9, beta_2=0.999, amsgrad=False            
        model.compile(optimizer=adamOptimizer,
            loss='mean_squared_error',
        )
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
        callbacksArray = [es]
        if(return_best):
            mc = ModelCheckpoint('best_1DConv.h5', monitor='val_loss', mode='min')
            callbacksArray = [es, mc]      
        history=model.fit(trainingSet, trainingSetLabels,
                                batch_size=batchSize,
                                shuffle=True,
                                epochs=epochsParam,                             
                                validation_data=(validationSet, validationSetLabels),
                                callbacks=callbacksArray,
                                verbose = vrbs,
                            )        
        if(return_best):
            best_model = load_model('best_1DConv.h5')           
        returnModel = model
        if(return_best):
            returnModel = best_model
        return [returnModel,min(history.history['val_loss']),len(history.history['val_loss']),history]
        
    def trainLstm(self, numOfLayers, numOfNeurons, trainingSet, trainingSetLabels, validationSet, validationSetLabels, printSummary = 1, vrbs = 1, return_best = 1, learningRate = 0.001, epochsParam = 10000, batchSize = 5000):                
        print('Training model.')
        model = Sequential()
        model.add(LSTM(numOfNeurons, input_shape=(self.sequenceLen, self.dimensionsCount), return_sequences=True))
        for j in range(0, numOfLayers - 1):
            model.add(LSTM(numOfNeurons, return_sequences=True))
        model.add(Dense(self.dimensionsCount))
        model.add(Activation("linear"))
        if(printSummary == True):
            model.summary()
        adamOptimizer = Adam(learning_rate=learningRate) # , beta_1=0.9, beta_2=0.999, amsgrad=False            
        model.compile(optimizer=adamOptimizer,
            loss='mean_squared_error',
        )
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
        callbacksArray = [es]
        if(return_best):
            mc = ModelCheckpoint('best_lstm.h5', monitor='val_loss', mode='min')
            callbacksArray = [es, mc]
        history=model.fit(trainingSet, trainingSetLabels,
                            batch_size=batchSize,
                            shuffle=True,
                            epochs=epochsParam, 
                            validation_data=(validationSet, validationSetLabels),
                            callbacks=callbacksArray,
                            verbose = vrbs,
                        )
        if(return_best):
            best_model = load_model('best_lstm.h5')
        returnModel = model
        if(return_best):
            returnModel = best_model
        return [returnModel,min(history.history['val_loss']),len(history.history['val_loss']),history]
        
    def testModel(self, testSet, testSetLabels, modelType, vrbs = 1):
        testSet = getReshapedDataSetNoSplit(testSet, modelType)
        precisions = []
        recalls = []
        model = load_model('Trained_Model/' + modelType + '.h5')
        predicted = model.predict(testSet)
        mse = None
        if(modelType == "lstm"):
            mse = (np.square(testSet - predicted)).mean(axis=2).mean(axis=1)
        else:
            mse = (np.square(testSet - predicted)).mean(axis=1)
        if(modelType == "conv"):
            mse = mse.reshape(testSet.shape[0])
        mse_label = np.vstack((mse, testSetLabels)).T
        precision, recall, minPositiveMSE, maxNegativeMSE  = rankedPrecisionAndRecall(mse_label)    
        precisions.append(precision)
        recalls.append(recall)
        thre = calculateThreshold(minPositiveMSE,maxNegativeMSE)
        precisionThre, recallThre = rankedPrecisionAndRecallWithThreshold(mse_label,thre)
        if(vrbs == 1):
            print("Precision without threshold is: ", precision)
            print("Recall without threshold is: ", precision)
            print("Min reconstruction error for anomalies: ", minPositiveMSE) # 0.010118610983828593
            print("max reconstruction error for benign: ", maxNegativeMSE) 
            print("Selected threshold: ",thre)
            print("Precision by threshold is: ", precisionThre)
            print("Recall by threshold is: ", recallThre)
        return precision, recall, precisionThre, recallThre, thre, minPositiveMSE, maxNegativeMSE

