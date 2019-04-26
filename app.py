import numpy as np
from IPython.display import Markdown, display


class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        # this is how it'll look like
        # [number,
        #  number,
        #  number]
        self.synaptic_weights = 2 * np.random.random((5, 1)) - 1

    def sigmoid(self, x):
        # applying the sigmoid function to calculate the activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron

            # 1. Forward propagation
            output = self.predict(training_inputs)

            # 2. Back propagation
            self.synaptic_weights += self.BackPropagation(
                output, training_inputs, training_outputs)

    def BackPropagation(self, output, training_inputs, training_outputs):

        # computing error rate for back-propagation
        error = training_outputs - output

        # performing weight adjustments
        adjustments = np.dot(training_inputs.T, error *
                             self.sigmoid_derivative(output))

        return adjustments

    def predict(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def printmd(self, msg):
        display(Markdown(msg))


if __name__ == "__main__":

    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # training data consisting of 4 examples--3 features and 1 output
    training_inputs = np.array([[0, 1, 1, 1, 0],
                                [1, 1, 0, 0, 0],
                                [1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [1, 1, 0, 0, 1],
                                [0, 0, 1, 1, 0],
                                [0, 1, 0, 1, 0],
                                [1, 1, 0, 1, 1]])

    # the class vector has to be column vector so we used T to transpose it
    # 0 --> Cat  | 1 --> Dog
    training_outputs = np.array([[0, 0, 1, 1, 1, 0, 0, 1]]).T

    # training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    neural_network.printmd("**0 --> No\n 1 --> yes**")
    user_input_one = str(input("Is it big? "))
    user_input_two = str(input("Is it sensetive?  "))
    user_input_three = str(input("Does it like weird places? "))
    user_input_four = str(input("Does it give birth alot? "))
    user_input_five = str(input("Does it guard your house?"))

    print("Your features are: ", user_input_one,
          user_input_two, user_input_three, user_input_four, user_input_five)

    result = neural_network.predict(
        np.array([user_input_one, user_input_two, user_input_three,
                  user_input_four, user_input_five]))

    result = round(result[0])

    print("The output is: ")
    if result == 0:
        print("it's a Cat.")
    else:
        print("it's a Dog.")
