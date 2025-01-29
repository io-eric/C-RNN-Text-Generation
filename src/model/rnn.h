#pragma once
#pragma once
#include "../matrix/matrix.h"
#include "../vocabulary/vocabulary.h"

typedef struct
{
    int input_size;         // Size of the input vector (e.g., vocabulary size)
    int hidden_size;        // Size of the hidden state vector
    int output_size;        // Size of the output vector (e.g., vocabulary size)
    double learning_rate;   // Learning rate for training
    Matrix *hidden_weights; // Weights for the hidden state (input to hidden)
    Matrix *output_weights; // Weights for the output (hidden to output)
    Matrix *hidden_state;   // Current hidden state of the RNN
} RNN;

RNN *rnn_init(int input_size, int hidden_size, int output_size, double learning_rate);
void rnn_free(RNN *rnn);
Matrix *rnn_forward(RNN *rnn, Matrix *input);               // Forward pass
void rnn_backward(RNN *rnn, Matrix *input, Matrix *target); // Backward pass (backpropagation through time)
char *rnn_generate_text(Vocabulary *v, RNN *rnn, char *initial_input, int length);
void rnn_save(RNN *rnn, const char *filename);
RNN *rnn_load(const char *filename);