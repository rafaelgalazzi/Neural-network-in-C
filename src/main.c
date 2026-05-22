#include "neural_network/inferenceModel.h"
#include <stdio.h>
#include <stdlib.h>

// compile with
// gcc src\main.c src\neural_network\inferenceModel.c src\random_number_generator\randomNumber.c -I src -o xor_demo.exe

int main()
{
    size_t networkShape[] = {2, 8, 20, 1};
    size_t numberOfLayers = sizeof(networkShape) / sizeof(networkShape[0]);
    size_t inputSize = networkShape[0];
    size_t outputSize = networkShape[numberOfLayers - 1];

    double inputRange = 0.5;
    NeuralNetwork network = createNeuralNetwork(networkShape, numberOfLayers, inputRange);
    if (!network.weights || !network.bias)
    {
        return 1;
    }

    double trainingInputs[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0};

    double targetOutputs[] = {
        0.0,
        1.0,
        1.0,
        0.0};

    size_t numberOfSamples = 4;
    size_t epochs = 50000;
    double learningRate = 0.1;

    // loadModel(&network, "model.txt");

    trainModel(&network, trainingInputs, targetOutputs, numberOfSamples, epochs, learningRate);

    saveModel(&network, "model.txt");

    size_t totalNeurons = countTotalNeurons(&network);
    if (!totalNeurons)
    {
        printf("Allocation memory failt to totalNeurons");
        return -1;
    }
    double *allNeuronValues = (double *)malloc(totalNeurons * sizeof(double));
    if (!allNeuronValues)
    {
        printf("Allocation memory fail to allNeuronValues");
        freeNeuralNetwork(&network);
        return 1;
    }

    double *output = (double *)malloc(outputSize * sizeof(double));
    if (!output)
    {
        printf("Allocation memory fail to output");
        free(allNeuronValues);
        freeNeuralNetwork(&network);
        return 1;
    }

    printf("XOR predictions after training:\n");

    for (size_t sample = 0; sample < numberOfSamples; sample++)
    {
        double *input = trainingInputs + (sample * inputSize);

        forwardPass(&network, input, output, allNeuronValues);

        printf("%.0f XOR %.0f = %.4f (expected %.0f)\n", input[0], input[1], output[0], targetOutputs[sample]);
    }

    free(output);
    freeNeuralNetwork(&network);
    free(allNeuronValues);
    return 0;
}
