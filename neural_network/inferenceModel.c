#include <stdint.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "randomNumber.h"
#include "inferenceModel.h"

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

double activationFunction(double value)
{
    return 1.0f / (1.0f + exp(-value));
}

double activationDerivate(double value)
{
    double s = activationFunction(value);
    return s * (1 - s);
}

// Create a neural network based on an integer array passed the first element will be the input, the last the output
NeuralNetwork createNeuralNetwork(int *networkShape, size_t numberOfLayers)
{
    NeuralNetwork network = {0};
    network.numberOfLayers = numberOfLayers;
    network.networkShape = networkShape;

    // Allocating memory for weights
    int numberOfConnections = 0;
    for (int i = 0; i < numberOfLayers - 1; i++)
    {
        numberOfConnections += *(networkShape + i) * (*(networkShape + i + 1));
    }

    network.weights = malloc(numberOfConnections * sizeof(double));

    if (!network.weights)
    {
        printf("Allocation memory fail");
        return network;
    }

    // Allocating memory for bias
    int numberOfElementsWithBias = 0;
    for (int i = 0; i < numberOfLayers - 1; i++)
    {
        numberOfElementsWithBias += *(networkShape + i + 1);
    }

    network.bias = malloc(numberOfElementsWithBias * sizeof(double));
    if (!network.bias)
    {
        printf("Allocation memory fail");
        free(network.weights);
        return network;
    }

    // Initializating values on bias
    for (int i = 0; i < numberOfElementsWithBias; i++)
    {
        *(network.bias + i) = randomNumber(-10, 10);
    }

    // Initializating values on weights
    for (int i = 0; i < numberOfConnections; i++)
    {
        *(network.weights + i) = randomNumber(-10, 10);
    }

    return network;
}

void forwardPass(NeuralNetwork *network, double *input, double *output)
{
    if (!network || !network->networkShape || !network->weights || !network->bias || !input || !output)
    {
        return;
    }

    size_t numberOfInputElements = *(network->networkShape);
    size_t currentNumberofElementsInInput = numberOfInputElements;
    double *currentLayerElements = malloc(numberOfInputElements * sizeof(double));

    if (!currentLayerElements)
    {
        printf("Allocation memory fail");
        return;
    }

    for (size_t i = 0; i < numberOfInputElements; i++)
    {
        *(currentLayerElements + i) = *(input + i);
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

        double *resized = realloc(currentLayerElements, numberOfElementsNextLayer * sizeof(double));
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

void backwardPass(NeuralNetwork *network, double *output)
{
}

void trainModel(NeuralNetwork *network)
{
}

void freeNeuralNetwork(NeuralNetwork *network)
{
    free(network->weights);
    free(network->bias);
    network->weights = NULL;
    network->bias = NULL;
}