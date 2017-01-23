from neuralnet import NeuralNetwork
from numpy import exp, array, random, dot
import numpy as np
import json
from random import shuffle
words_from_every_lang = {}

with open('./words.json', 'r+') as words:
    json_words = json.load(words)
    verbs = json_words['verb']
    nouns = json_words['noun']
# Format of word description lists
# [vowel percent, consonant percent]

class VerbNeuralNet(NeuralNetwork):
    def __init__(self, synapse_count):
        random.seed(0)
        self.synaptic_weights = 2 * random.random((synapse_count,1)) - 1
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
        print(self.synaptic_weights)
class LanguageBrain():
    def __init__(self, synapse_count, names=False, neuron_count=0):
        self.synapse_count = synapse_count
        self.names = []
        if names == False:
            self.neurons = []
            self.grow(neuron_count)
        else:
            self.neurons = {}
            self.grow(names=names)
    def __str__(self):
        string = ""
        for neuron_num, neuron in self.neurons:
            string += "#########\n"
            string += "Neuron " + str(neuron_num) + " synapse weights \n"
            string += str(neuron.synaptic_weights) + "\n"
        return string
    def grow(self, names=False, neuron_count=0):
        if names == False:
            for x in range(neuron_count):
                self.neurons.append(VerbNeuralNet(self.synapse_count))
        else:
            self.names.extend(names)
            for neuron_name in names:
                print(neuron_name)
                self.neurons[neuron_name] = VerbNeuralNet(self.synapse_count)
    def learn(self, inputs, outputs, iterations):
        # things should be a list of lists hopefully
        # thing = [[inputs, outputs],[inputs, outputs], . . .]
        if len(outputs) < len(self.neurons):
            print("WARNING! Some neurons will not learn anything!")
        elif len(outputs) > len(self.neurons):
            raise ValueError("Amount of outputs to learn against greater than amount of neurons, must grow brain first")

        if self.names == False:
            for num, thing in things:
                self.neurons[num].train(iterations, things[0], thing[1])
        else:
            for name in self.names:
                print(name)
                self.neurons[name].train(iterations, inputs, outputs[name])
    def predict(self, inputs):
        results = {}
        if self.names == False:
            for neuron_num, neuron in enumerate(self.neurons):
                results[neuron_num] = neuron.predict(inputs)
        else:
            for key, neuron in self.neurons.items():
                results[key] = neuron.predict(inputs)
        return results


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
    minLength = 5
    filename = "data/" + filename
    filecontents = []
    with open(filename) as words:
        for line in words:
            if len(line.split(",")[0]) >= 5:
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
    words_from_every_lang["random"] = grabWords("output0.txt")
    words_from_every_lang["key mash"] = grabWords("output1.txt")
    words_from_every_lang["english"] = grabWords("output2.txt")
    words_from_every_lang["spanish"] = grabWords("output3.txt")
    words_from_every_lang["french"] = grabWords("output4.txt")
    words_from_every_lang["german"] = grabWords("output5.txt")
    words_from_every_lang["japanese"] = grabWords("output6.txt")
    words_from_every_lang["swahili"] = grabWords("output7.txt")
    words_from_every_lang["mandarin"] = grabWords("output8.txt")
    words_from_every_lang["esperanto"] = grabWords("output9.txt")
    words_from_every_lang["dutch"] = grabWords("output10.txt")
    words_from_every_lang["polish"] = grabWords("output11.txt")
    words_from_every_lang["lojban"] = grabWords("output12.txt")

    word_descriptors = []
    words_to_train_on = []
    training_set_outputs = {}
    languages_to_train = ["english","mandarin"]
    training_input = []
    training_outputs = {}
    # initialize training_outputs
    for lang in languages_to_train:
        training_outputs[lang] = []
        for word in words_from_every_lang[lang]:
            words_to_train_on.append(word)

    shuffle(words_to_train_on)
    # define training inputs and outputs more thoroughly

    for word in words_to_train_on:
        descriptor = genDescriptor(word)
        training_input.append(descriptor)
        for lang in languages_to_train:
            if word in words_from_every_lang[lang]:
                training_outputs[lang].append([1])
            else:
                training_outputs[lang].append([0])


    # grow a brain with 13 neurons and 390 synapses for every neuron
    brain = LanguageBrain(26*15, names=languages_to_train)


    for key, value in training_outputs.items():
        training_set_outputs[key] = array(value)
    training_input = array(training_input)

    brain.learn(training_input, training_set_outputs, 10000)

    while True:
        word = str(input("Enter a word: "))
        results = brain.predict(genDescriptor(word))
        for lang in languages_to_train:
            print(str(results[lang]) + " probability of being " + str(lang))
