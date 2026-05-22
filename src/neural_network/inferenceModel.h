#ifndef INFERENCEMODEL_H
#define INFERENCEMODEL_H
#include <stddef.h>

typedef struct
{
    double *weights;
    // Array linear with all values of the weights number total of connections

    double *bias;
    // Bias for each layer, will be added with the input for the next layer and passed inside the non linear function

    size_t *networkShape;
    // each element represent the number of neurons of each layer including input and output
    // size: numberOfLayers

    size_t numberOfLayers;
} NeuralNetwork;

double activationFunction(double value);
double activationDerivate(double value);

size_t countTotalNeurons(NeuralNetwork *network);

NeuralNetwork createNeuralNetwork(size_t *networkShape, size_t numberOfLayers);

// forwardPass responsable for making the inference
void forwardPass(NeuralNetwork *network, double *input, double *output, double *allNeuronValues);

// backwardPass(backpropagation) responsable for training the model by adjusting the weights
void backwardPass(NeuralNetwork *network, double *input, double *output, double *targetOutput, double *allNeuronValues, double learningRate);

double calculateLoss(double output, double expected);

double calculateDerivateLoss(double output, double expected);

void trainModel(NeuralNetwork *network, double *trainingInputs, double *targetOutputs, size_t numberOfSamples, size_t epochs, double learningRate);

void freeNeuralNetwork(NeuralNetwork *network);

#endif
