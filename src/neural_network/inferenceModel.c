#include <stdint.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "random_number_generator/randomNumber.h"
#include "neural_network/inferenceModel.h"

/*
    Uma rede neural pode ser construidas fazendo a definição do problema e como deve ser a resposta esperada
    No caso abaixo inserimos o tabuleiro com a condição atual e de que é o turno e a resposta é um vetor que representa a mesa
    e a proxima jogada é o elemento com maior probabilidade.


    Input = {-1, 0, 0, 1, -1, 0, 0, 0, 1, 1}

    Hidden Layers = 3 x 10 (3 camadas de 10, é arbitrária mas seu tamanho depende do problema)

    Output = {0, 0 , 0.95, 0, 0, 0, 0, 0, 0}

    Output = Ativação()

    In => H1 => H2 => H3 => Out

    Cada Input tem uma ligação com cada neurônio da proxima camada, ligados pela matriz de pesos;


    Numero de pesos na W1 = Quantidade de Inputs X Quantidade de neuronios na cama oculta 1
    W1 = 10 x 10 = 100

    Pesos W2 = 10 x 10 = 100

    Pesos W3 = 10 x 10 = 100

    Pesos Wout = 10 x 9 = 90

    Saida = AtOut(Wout X OutputW3 + BiosOut);

*/

// numero de inputs x numero de elementos proxima camada x camadas

double activationFunction(double preActivationValue)
{
    return 1.0f / (1.0f + exp(-preActivationValue));
}

double activationDerivate(double output)
{
    return output * (1 - output);
}

double calculateLoss(double output, double expected)
{
    double error = output - expected;
    return error * error / 2.0;
}

double calculateDerivateLoss(double output, double expected)
{
    return output - expected;
}

size_t countTotalWeights(NeuralNetwork *network)
{
    size_t totalWeights = 0;
    for (size_t i = 0; i < network->numberOfLayers - 1; i++)
    {
        size_t neuronsInCurrentLayer = *(network->networkShape + i);
        size_t neuronsInNextLayer = *(network->networkShape + i + 1);
        totalWeights += neuronsInCurrentLayer * neuronsInNextLayer;
    }

    return totalWeights;
}

size_t countTotalBias(NeuralNetwork *network)
{
    size_t totalNeurons = 0;
    for (size_t i = 1; i < network->numberOfLayers; i++)
    {
        totalNeurons += *(network->networkShape + i);
    }

    return totalNeurons;
}

size_t countTotalNeurons(NeuralNetwork *network)
{
    size_t totalNeurons = 0;
    for (size_t i = 0; i < network->numberOfLayers; i++)
    {
        totalNeurons += *(network->networkShape + i);
    }

    return totalNeurons;
}

size_t neuronLayerOffset(NeuralNetwork *network, size_t layer)
{
    size_t offset = 0;
    for (size_t i = 0; i < layer; i++)
    {
        offset += *(network->networkShape + i);
    }

    return offset;
}

size_t weightLayerOffset(NeuralNetwork *network, size_t layer)
{
    size_t offset = 0;
    for (size_t i = 0; i < layer; i++)
    {
        offset += *(network->networkShape + i) * (*(network->networkShape + i + 1));
    }

    return offset;
}

size_t biasLayerOffset(NeuralNetwork *network, size_t layer)
{
    size_t offset = 0;
    for (size_t i = 0; i < layer; i++)
    {
        offset += *(network->networkShape + i + 1);
    }

    return offset;
}

void saveModel(NeuralNetwork *network, char *modelName)
{
    if (!network || !network->networkShape || !network->weights || !network->bias || !modelName)
    {
        return;
    }

    FILE *model = fopen(modelName, "w");
    if (!model)
    {
        printf("Fail while opening the file %s", modelName);
        return;
    }

    size_t amountOfWeights = countTotalWeights(network);
    size_t amountOfBias = countTotalBias(network);

    fprintf(model, "Number of layers\n");

    fprintf(model, "%zu\n", network->numberOfLayers);

    fprintf(model, "Network Shape\n");

    for (size_t i = 0; i < network->numberOfLayers; i++)
    {
        fprintf(model, "%zu\n", *(network->networkShape + i));
    }

    fprintf(model, "Weights\n");

    for (size_t i = 0; i < amountOfWeights; i++)
    {
        fprintf(model, "%.17g\n", *(network->weights + i));
    }

    fprintf(model, "Bias\n");

    for (size_t i = 0; i < amountOfBias; i++)
    {
        fprintf(model, "%.17g\n", *(network->bias + i));
    }

    if (fclose(model) != 0)
    {
        printf("Error while closing file!");
        return;
    }
}

void loadModel(NeuralNetwork *network, char *modelName)
{
    if (!network || !network->networkShape || !network->weights || !network->bias || !modelName)
    {
        return;
    }

    FILE *model = fopen(modelName, "r");

    if (!model)
    {
        printf("Error while opening file %s", modelName);
        return;
    }

    size_t numberOfLayers = 0;
    if (fscanf(model, "Number of layers %zu Network Shape", &numberOfLayers) != 1)
    {
        printf("Error while reading number of layers");
        fclose(model);
        return;
    }

    if (numberOfLayers != network->numberOfLayers)
    {
        printf("Model layer count does not match the current network");
        fclose(model);
        return;
    }

    for (size_t i = 0; i < numberOfLayers; i++)
    {
        size_t shapeElement = 0;
        if (fscanf(model, "%zu", &shapeElement) != 1)
        {
            printf("Error while reading network shape");
            fclose(model);
            return;
        }

        if (shapeElement != *(network->networkShape + i))
        {
            printf("Model shape does not match the current network");
            fclose(model);
            return;
        }
    }

    size_t amountOfWeights = countTotalWeights(network);
    size_t amountOfBias = countTotalBias(network);

    fscanf(model, " Weights");
    for (size_t i = 0; i < amountOfWeights; i++)
    {
        if (fscanf(model, "%lf", network->weights + i) != 1)
        {
            printf("Error while reading weights");
            fclose(model);
            return;
        }
    }

    fscanf(model, " Bias");
    for (size_t i = 0; i < amountOfBias; i++)
    {
        if (fscanf(model, "%lf", network->bias + i) != 1)
        {
            printf("Error while reading bias");
            fclose(model);
            return;
        }
    }

    fclose(model);
}

// Create a neural network based on an integer array passed the first element will be the input, the last the output
NeuralNetwork createNeuralNetwork(size_t *networkShape, size_t numberOfLayers, double inputRange)
{
    NeuralNetwork network = {0};
    network.numberOfLayers = numberOfLayers;
    network.networkShape = networkShape;

    // Allocating memory for weights
    size_t numberOfConnections = 0;
    for (size_t i = 0; i < numberOfLayers - 1; i++)
    {
        numberOfConnections += *(networkShape + i) * (*(networkShape + i + 1));
    }

    network.weights = (double *)malloc(numberOfConnections * sizeof(double));

    if (!network.weights)
    {
        printf("Allocation memory fail");
        return network;
    }

    // Allocating memory for bias
    size_t numberOfElementsWithBias = 0;
    for (size_t i = 0; i < numberOfLayers - 1; i++)
    {
        numberOfElementsWithBias += *(networkShape + i + 1);
    }

    network.bias = (double *)malloc(numberOfElementsWithBias * sizeof(double));
    if (!network.bias)
    {
        printf("Allocation memory fail");
        free(network.weights);
        return network;
    }

    // Initializating values on bias
    for (size_t i = 0; i < numberOfElementsWithBias; i++)
    {
        *(network.bias + i) = randomNumber(-inputRange, inputRange);
    }

    // Initializating values on weights
    for (size_t i = 0; i < numberOfConnections; i++)
    {
        *(network.weights + i) = randomNumber(-inputRange, inputRange);
    }

    return network;
}

void forwardPass(NeuralNetwork *network, double *input, double *output, double *allNeuronValues)
{
    if (!network || !network->networkShape || !network->weights || !network->bias || !input || !output || !allNeuronValues)
    {
        return;
    }

    size_t numberOfInputElements = *(network->networkShape);
    size_t currentNumberofElementsInInput = numberOfInputElements;
    double *currentLayerElements = (double *)malloc(numberOfInputElements * sizeof(double));

    if (!currentLayerElements)
    {
        printf("Allocation memory fail");
        return;
    }
    size_t neuronCount = 0;
    for (size_t i = 0; i < numberOfInputElements; i++)
    {
        *(currentLayerElements + i) = *(input + i);
        *(allNeuronValues + neuronCount) = *(input + i);
        neuronCount++;
    }

    size_t weightCount = 0;
    size_t biasCount = 0;
    for (size_t i = 0; i < network->numberOfLayers - 1; i++)
    {
        size_t numberOfElementsNextLayer = *(network->networkShape + i + 1);

        double *nextLayerOutput = (double *)malloc(numberOfElementsNextLayer * sizeof(double));

        if (!nextLayerOutput)
        {
            printf("Allocation memory fail");
            free(currentLayerElements);
            return;
        }

        // Wi(Current weight matrix) X Ii(Current input matrix) + Bias(Current Bias)
        for (size_t nextLayerInputElement = 0; nextLayerInputElement < numberOfElementsNextLayer; nextLayerInputElement++)
        {
            double lineSum = 0;
            for (size_t inputElement = 0; inputElement < currentNumberofElementsInInput; inputElement++)
            {
                lineSum += *(network->weights + weightCount) * (*(currentLayerElements + inputElement));
                weightCount++;
            }

            *(nextLayerOutput + nextLayerInputElement) = activationFunction(lineSum + *(network->bias + biasCount));
            biasCount++;
        }

        double *resized = (double *)realloc(currentLayerElements, numberOfElementsNextLayer * sizeof(double));
        if (!resized)
        {
            free(currentLayerElements);
            free(nextLayerOutput);
            return;
        }

        currentLayerElements = resized;
        currentNumberofElementsInInput = numberOfElementsNextLayer;

        for (size_t nextLayerInputElement = 0; nextLayerInputElement < numberOfElementsNextLayer; nextLayerInputElement++)
        {
            *(currentLayerElements + nextLayerInputElement) = *(nextLayerOutput + nextLayerInputElement);
            *(allNeuronValues + neuronCount) = *(nextLayerOutput + nextLayerInputElement);
            neuronCount++;
        }

        free(nextLayerOutput);
    }

    size_t numberOfOutputElements = *(network->networkShape + network->numberOfLayers - 1);
    for (size_t i = 0; i < numberOfOutputElements; i++)
    {
        *(output + i) = *(currentLayerElements + i);
    }

    free(currentLayerElements);
}

// ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w
// w (new) = w(old) - n * ∂L/∂w
void backwardPass(NeuralNetwork *network, double *input, double *output, double *targetOutput, double *allNeuronValues, double learningRate)
{
    // Validate the pointers used by backpropagation before calculating gradients
    if (!network || !network->networkShape || !network->weights || !network->bias || !input || !output || !targetOutput || !allNeuronValues)
    {
        return;
    }

    // Calculate the loss to know how far the current prediction is from the expected output
    double totalLoss = 0;
    size_t numberOfOutputElements = *(network->networkShape + network->numberOfLayers - 1);
    for (size_t i = 0; i < numberOfOutputElements; i++)
    {
        double outputValue = *(output + i);
        double targetOutputValue = *(targetOutput + i);
        totalLoss += calculateLoss(outputValue, targetOutputValue);
    }

    // calculate the delta for every neuron
    // Deltas represent how much each neuron contributed to the final error
    size_t numberOfNeuronsToCalculateDelta = 0;
    for (size_t i = 1; i < network->numberOfLayers; i++)
    {
        numberOfNeuronsToCalculateDelta += *(network->networkShape + i);
    }

    // Store the delta of every neuron using the same layer order used by allNeuronValues
    size_t totalNeurons = countTotalNeurons(network);
    double *deltas = (double *)malloc(totalNeurons * sizeof(double));
    if (!deltas)
    {
        printf("Allocation memory for deltas fail");
        return;
    }

    for (size_t i = 0; i < totalNeurons; i++)
    {
        *(deltas + i) = 0;
    }

    // output delta calculateDerivateLoss * activationDerivate
    // The output layer delta starts the chain rule because it compares prediction with target
    double *outputDeltas = (double *)malloc(numberOfOutputElements * sizeof(double));
    if (!outputDeltas)
    {
        printf("Allocation memory for outputDeltas fail");
        free(deltas);
        return;
    }

    size_t outputLayerOffset = neuronLayerOffset(network, network->numberOfLayers - 1);
    for (size_t i = 0; i < numberOfOutputElements; i++)
    {
        double outputValue = *(output + i);
        double targetOutputValue = *(targetOutput + i);
        *(outputDeltas + i) = calculateDerivateLoss(outputValue, targetOutputValue) * activationDerivate(outputValue);
        *(deltas + outputLayerOffset + i) = *(outputDeltas + i);
    }

    double *hiddenDeltas = (double *)malloc(numberOfNeuronsToCalculateDelta * sizeof(double));
    if (!hiddenDeltas)
    {
        printf("Allocation memory for hiddenDeltas fail");
        free(outputDeltas);
        free(deltas);
        return;
    }

    for (size_t i = 0; i < numberOfNeuronsToCalculateDelta; i++)
    {
        *(hiddenDeltas + i) = 0;
    }

    // calculate deltas for hidden layers neurons
    // Hidden deltas are calculated backwards using the next layer deltas and the connecting weights
    for (size_t layer = network->numberOfLayers - 2; layer > 0; layer--)
    {
        size_t numberOfElementsOnLayer = *(network->networkShape + layer);
        size_t numberOfElementsOnNextLayer = *(network->networkShape + layer + 1);
        size_t currentLayerOffset = neuronLayerOffset(network, layer);
        size_t nextLayerOffset = neuronLayerOffset(network, layer + 1);
        size_t currentWeightLayerOffset = weightLayerOffset(network, layer);

        for (size_t currentNeuron = 0; currentNeuron < numberOfElementsOnLayer; currentNeuron++)
        {
            double nextLayerDeltasSum = 0;
            for (size_t nextNeuron = 0; nextNeuron < numberOfElementsOnNextLayer; nextNeuron++)
            {
                size_t weightIndex = currentWeightLayerOffset + (nextNeuron * numberOfElementsOnLayer) + currentNeuron;
                nextLayerDeltasSum += *(network->weights + weightIndex) * (*(deltas + nextLayerOffset + nextNeuron));
            }

            *(deltas + currentLayerOffset + currentNeuron) = nextLayerDeltasSum * activationDerivate(*(allNeuronValues + currentLayerOffset + currentNeuron));
        }
    }

    // Keep a compact hiddenDeltas array with all non-input neuron deltas
    for (size_t i = 0; i < numberOfNeuronsToCalculateDelta; i++)
    {
        *(hiddenDeltas + i) = *(deltas + *(network->networkShape) + i);
    }

    // Apply gradient descent to every weight and bias after all deltas were calculated
    for (size_t layer = 0; layer < network->numberOfLayers - 1; layer++)
    {
        size_t numberOfElementsOnLayer = *(network->networkShape + layer);
        size_t numberOfElementsOnNextLayer = *(network->networkShape + layer + 1);
        size_t currentLayerOffset = neuronLayerOffset(network, layer);
        size_t nextLayerOffset = neuronLayerOffset(network, layer + 1);
        size_t currentWeightLayerOffset = weightLayerOffset(network, layer);
        size_t currentBiasLayerOffset = biasLayerOffset(network, layer);

        for (size_t nextNeuron = 0; nextNeuron < numberOfElementsOnNextLayer; nextNeuron++)
        {
            double delta = *(deltas + nextLayerOffset + nextNeuron);
            for (size_t currentNeuron = 0; currentNeuron < numberOfElementsOnLayer; currentNeuron++)
            {
                // Weight gradient = next neuron delta * current neuron activation
                size_t weightIndex = currentWeightLayerOffset + (nextNeuron * numberOfElementsOnLayer) + currentNeuron;
                double currentNeuronValue = *(allNeuronValues + currentLayerOffset + currentNeuron);
                *(network->weights + weightIndex) -= learningRate * delta * currentNeuronValue;
            }

            // Bias gradient is the delta itself because bias is added directly to z
            *(network->bias + currentBiasLayerOffset + nextNeuron) -= learningRate * delta;
        }
    }

    (void)totalLoss;
    free(hiddenDeltas);
    free(outputDeltas);
    free(deltas);
}

void trainModel(NeuralNetwork *network, double *trainingInputs, double *targetOutputs, size_t numberOfSamples, size_t epochs, double learningRate)
{
    // Validate the training data and network pointers before allocating training buffers
    if (!network || !network->networkShape || !network->weights || !network->bias || !trainingInputs || !targetOutputs)
    {
        return;
    }

    // Read the input/output size from the network shape so the flat training arrays can be indexed
    size_t numberOfInputElements = *(network->networkShape);
    size_t numberOfOutputElements = *(network->networkShape + network->numberOfLayers - 1);
    size_t totalNeurons = countTotalNeurons(network);

    // Reuse one output buffer for every sample during training
    double *output = malloc(numberOfOutputElements * sizeof(double));
    if (!output)
    {
        printf("Allocation memory for train output fail");
        return;
    }

    // Store all activations from the forward pass because backpropagation needs them for gradients
    double *allNeuronValues = malloc(totalNeurons * sizeof(double));
    if (!allNeuronValues)
    {
        printf("Allocation memory for train allNeuronValues fail");
        free(output);
        return;
    }

    // Repeat the training set many times so weights move gradually toward lower loss
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // Each sample is one full forward pass followed by one backward pass
        for (size_t sample = 0; sample < numberOfSamples; sample++)
        {
            // Move through the flat input and target arrays using the sample index
            double *sampleInput = trainingInputs + (sample * numberOfInputElements);
            double *sampleTargetOutput = targetOutputs + (sample * numberOfOutputElements);

            // First calculate the current prediction, then adjust weights using the prediction error
            forwardPass(network, sampleInput, output, allNeuronValues);
            backwardPass(network, sampleInput, output, sampleTargetOutput, allNeuronValues, learningRate);
        }
    }

    free(output);
    free(allNeuronValues);
}

void freeNeuralNetwork(NeuralNetwork *network)
{
    free(network->weights);
    free(network->bias);
    network->weights = NULL;
    network->bias = NULL;
}
