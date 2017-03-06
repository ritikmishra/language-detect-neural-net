from neuron import NeuralNetwork
from numpy import exp, array, random, dot
import numpy as np
import json
from random import shuffle
words_from_every_lang = {}

class VerbNeuralNet(NeuralNetwork):
    def __init__(self, synapse_count, type):
        if type == 1:
            self.synaptic_weights = np.ones((synapse_count,1))
        elif type == 0:
            self.synaptic_weights = np.zeros((synapse_count,1))
        elif type == -1:
            self.synaptic_weights = -1 * np.ones((synapse_count,1))
        elif type == "rand":
            np.random.seed(1)
            self.synaptic_weights = 2 * np.random.random((synapse_count,1)) - 1
    def __sigmoid_derivative(self, x):
        # back propogation = propogating error back into neuron to adjust weights
        return exp(-x)/((1+exp(-x))**2)
    def train(self, iterations, training_inputs, training_outputs):
        print("training")
        self.inputs = training_inputs
        self.outputs = training_outputs


        for iteration in range(iterations):
            #pass training set through neuron
            output = self.predict(self.inputs)
            # calculate error
            error = self.outputs - output

            # multiply the error by the input and again by the gradient of the sigmoid
            # aka gradient descent
            adjustment = dot(training_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment
            if iteration % iterations/10 == 0:
                print("Generation " + str(iteration))
        

def genVerbDescriptor(word):

    vowels = ['a', 'e', 'i', 'o', 'u']
    verb_suffixes = ['ing', 'ed', 'ate', 'ify', 'en', 'ize', 'ise']

    vowel_num = 0
    length = len(word)
    consonant_num = 0
    descriptor = []

    for letter in word:
        if letter.lower() in vowels:
            vowel_num += 1
        elif letter != " " and letter != "-":
            consonant_num += 1

    descriptor.append(vowel_num)
    descriptor.append(consonant_num)

    for suffix in verb_suffixes:
        if word[-1 * len(suffix):].lower() == suffix:
            descriptor.append(1)
            break
    try:
        descriptor[2]
    except IndexError:
        descriptor.append(0)
    return descriptor
def grabWords(filename):
    filecontents = []
    with open(filename) as words:
        for line in words:
            filecontents.append(line.split(",")[0].lower())
    return filecontents
def genDescriptor(word):
    word = word[0:15]
    if len(word) < 15:
        for x in range(15-len(word)):
            word += " "
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    descriptor = []
    base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for letter in word:
        letter_descriptor = base
        for pos, char in enumerate(alphabet):
            if letter == char:
                letter_descriptor[pos] = 1
        descriptor.extend(letter_descriptor)
    return descriptor


if __name__ == "__main__":
    
    english = grabWords("data/output2.txt")
    random = grabWords("data/output0.txt")


    inputs = grabWords("inputs.txt")
    outputs = grabWords("outputs.txt")
    
    test_inputs = []
    test_outputs = outputs

    for word in inputs:
        test_inputs.append(genDescriptor(word))
    
    
    word_descriptors = []
    words_to_train_on = english + random
    training_set_outputs = {}
    training_input = []
    training_outputs = []
    
    shuffle(words_to_train_on)
    
    for word in words_to_train_on:
        descriptor = genDescriptor(word)
        training_input.append(descriptor)
        if word in english:
            training_outputs.append([1])
        else:
            training_outputs.append([0])
    
    
    training_outputs = array(training_outputs)
    training_input = array(training_input)
    
    print(training_outputs.shape)
    print(training_input.shape)
    
    all_one = VerbNeuralNet(390, 1)
    all_zero = VerbNeuralNet(390, 0)
    all_negone = VerbNeuralNet(390, -1)
    all_rand = VerbNeuralNet(390, "rand")

    trainings = 10000

    all_one.train(trainings, training_input, training_outputs)
    all_zero.train(trainings, training_input, training_outputs)
    all_negone.train(trainings, training_input, training_outputs)
    all_rand.train(trainings, training_input, training_outputs)
    
    all_one_err = 0.0
    all_zero_err = 0.0
    all_negone_err = 0.0
    all_rand_err = 0.0
    print(test_inputs)
    for x, word in enumerate(test_inputs):
        all_one_err += abs(float(all_one.predict(array(word))[0]) - float(test_outputs[x]))
        all_zero_err += abs(float(all_zero.predict(array(word))[0]) - float(test_outputs[x])) 
        all_negone_err += abs(float(all_negone.predict(array(word))[0]) - float(test_outputs[x]))
        all_rand_err += abs(float(all_rand.predict(array(word))[0]) - float(test_outputs[x]))
    print("one: " + str((100-all_one_err)/100))
    print("0: " + str((100-all_zero_err)/100))
    print("-1: " + str((100-all_negone_err)/100))
    print("rand: " + str((100-all_rand_err)/100))
