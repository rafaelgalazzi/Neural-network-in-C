#include <stdint.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
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

float activationFunction(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

float activationDerivate(float value)
{
    float s = activationFunction(value);
    return s * (1 - s);
}

// Create a neural network based on An integer array passed the first element will be the input, the last the output
NeuralNetwork createNeuralNetwork(int *networkShape, size_t numberOfLayers)
{
    NeuralNetwork network;
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
        free(network.weights);
        return;
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
        free(network.bias);
        return;
    }

    // Initializating values on weights
    int positionOnArray = 0;
    for (int i = 0; i < numberOfLayers - 1; i++)
    {
        for (int j = 0; j < *(networkShape + i); j++)
        {
            for (int k = 0; k < *(networkShape + i + 1); k++)
            {
                *(network.weights + positionOnArray) = randomNumber(-10, 10);
                positionOnArray++;
            }
        }
    }

    return network;
}

void forwardPass(NeuralNetwork network, double *input, size_t numberOfInputElements)
{

    size_t currentInputElementNumbers = numberOfInputElements;

    for (size_t i = 0; i < network.numberOfLayers - 1; i++)
    {
        int numberOfElementsNextLayer = *(network.networkShape + i + 1);

        double *nextLayerOutput = (double *)malloc(numberOfElementsNextLayer * sizeof(double));

        if (!nextLayerOutput)
        {
            printf("Allocation memory fail");
            free(nextLayerOutput);
            return;
        }

        for (size_t inputElement = 0; inputElement < currentInputElementNumbers; inputElement++) {
            
        }
    }
}

void backwardPass(NeuralNetwork network, double *output, size_t numberOfOutputElements)
{
}

void trainModel(NeuralNetwork network)
{
}