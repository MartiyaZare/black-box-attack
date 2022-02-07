from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM, Flatten, Reshape, ZeroPadding1D, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf
import random
import tqdm
from MZPackage.Utilities import *

class AdversarialGenerator:
    def __init__(self):
        config = loadConfig()        
        self.sequenceLen = config["sequenceLen"]
        self.dimensionsCount = config["dimensionsCount"]
        self.attackSamplesCount = config["attackSamplesCount"]        
        self.testSetBenignSize = config["testSetBenignSize"]
        self.thresholds = loadData("thresholds")


    def seqToFuncModel(self, model):
        input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            prev_layer = layer(prev_layer)

        funcModel = models.Model([input_layer], [prev_layer])
        return funcModel

    @tf.function # To fix: 'Tensor' object has no attribute 'numpy' 
    def create_adversarial_pattern(self, model, input_image, input_label):
        funcModel = self.seqToFuncModel(model)
        loss_object = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = funcModel(input_image)
            #loss = loss_object(input_label, prediction)
            loss = loss_object(input_image, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad
    
    @tf.function 
    def mzAttackSampleGeneratorFromBenign(self, model, iterations,seed,epsilon):
        tfInput = tfoutput = seed
        for step in range(0,iterations):
            tfInput = seed
            tfOutput = tfInput
            sgrad = self.create_adversarial_pattern(model, tfInput, tfOutput)                
            epsilon = 0.001
            perturbations = epsilon*sgrad
            seed = seed + perturbations    
        attackSample = seed
        return attackSample
        
    @tf.function 
    def mzAttackSampleGeneratorFromAnomaly(self, model, iterations,seed,epsilon):
        tfInput = tfoutput = seed
        for step in range(0,iterations):
            tfInput = seed
            tfOutput = tfInput
            sgrad = self.create_adversarial_pattern(model, tfInput, tfOutput)                        
            perturbations = epsilon*sgrad
            seed = seed - perturbations 
            
        attackSample = seed    
        return attackSample
        
    def generateRandomDataSeq(self, modelType):
        return getReshapedDataSetNoSplit(np.random.rand(1,self.sequenceLen,self.dimensionsCount),modelType)
        
    def generateAttackSamples(self, modelType, count, thre = None, aggressive = 0):
        maxIterations = 2000
        if(thre == None):
            thre = self.thresholds[modelType]
        if(aggressive):
            thre = thre/32
        model = load_model('Trained_Model/' + modelType + '.h5')
        attackSamples = list()    
        for i in tqdm.tqdm(range(count)):        
            attackSample = self.generateRandomDataSeq(modelType)
            mse = 1000                                              
            iteration = 0
            while mse > thre:            
                attackSample = self.mzAttackSampleGeneratorFromAnomaly(model, 1, attackSample, 0.001)            
                predicted = model.predict(attackSample)
                if(modelType == "lstm"):
                    mse = (np.square(attackSample - predicted)).mean(axis=2).mean(axis=1)  
                else:
                    mse = (np.square(attackSample - predicted)).mean(axis=1) 
                iteration += 1
                if(iteration >= maxIterations):
                    attackSample = self.generateRandomDataSeq(modelType)
                    iteration = 0                                
            attackSamples.append(attackSample)
                
        return np.array(attackSamples)
            