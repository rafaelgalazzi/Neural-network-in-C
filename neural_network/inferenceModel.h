#ifndef INFERENCEMODEL_H
#define INFERENCEMODEL_H

typedef struct
{
    double *weights;
    // Array linear with all values of the weights number total of connections

    double *bias;
    // Bias for each layer, will be added with the input for the next layer and passed inside the non linear function

    int *networkShape;
    // each element represent the number of neurons of each layer including input and output
    // size: numberOfLayers

    int numberOfLayers;
} NeuralNetwork;

float activationFunction(float value);
float activationDerivate(float value);

NeuralNetwork createNeuralNetwork(int *networkShape, size_t numberOfLayers);

// forwardPass responsable for making the inference
void forwardPass(NeuralNetwork network, double *input, size_t numberOfElements);

// backwardPass(backpropagation) responsable for training the model by adjusting the weights
void backwardPass(NeuralNetwork network, double *output, size_t numberOfOutputElements);


void trainModel(NeuralNetwork network);

#endif