from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #seed RNG so same numbers generated every time
        random.seed(1)

        # model neuron with 3 input connections, 1 output connection
        # assign random weights to 3 x 1 matrix with values between -1 and 1
        # mean of values is 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1
    def __sigmoid(self, x):
        """describes s shaped curve. we pass the weighted sum of the inputs in to normmalize them between 0 and 1"""
        e = 2.71828
        return 1/(1+exp(-x))
    def __sigmoid_derivative(self, x):
        # back propogation = propogating error back into neuron to adjust weights
        return exp(-x)/((1+exp(-x))**2)
    def train(self, number_of_trainings, training_set_inputs, training_set_outputs):
        self.inputs = training_set_inputs
        self.outputs = training_set_outputs
        self.len_input = len(self.inputs[0])
        for inpt in self.inputs:
            if len(inpt) != self.len_input:
                raise ValueError("All inputs must be the same length!")

        self.synaptic_weights = 2 * random.random((3,1)) - 1
        for iteration in range(number_of_trainings):
            #pass training set through neuron
            output = self.predict(self.inputs)

            # calculate error
            error = self.outputs - output

            # multiply the error by the input and again by the gradient of the sigmoid
            # aka gradient descent

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment
    def predict(self, inputs):
        """Pass inputs through neuron"""
        return self.__sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == "__main__":
    # Training set
    training_set_inputs = array([[0,1,1],[1,0,1],[0,0,1],[1,1,1]])
    training_set_outputs = array([[0,1,0,1]]).T


    # make neuron
    neural_net = NeuralNetwork()

    print("Random starting synaptic weights")
    print(neural_net.synaptic_weights)

    #train
    neural_net.train(10000, training_set_inputs, training_set_outputs)

    print("New synaptic weights")
    print(neural_net.synaptic_weights)

    new_data = input("What is the new consideration? ").split(" ")
    for index, val in enumerate(new_data):
        new_data[index] = int(val)

    print("Considering " + str(new_data) + " output")
    print(neural_net.predict(array(new_data)))
