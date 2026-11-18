#pragma once

#include "../matrix/matrix.h"
#include "../vocabulary/vocabulary.h"

typedef struct
{
    int input_size;       // Size of the input vector (e.g., vocabulary size)
    int hidden_size;      // Size of the hidden state vector
    int output_size;      // Size of the output vector (e.g., vocabulary size)
    double learning_rate; // Learning rate for training

    Matrix *hidden_weights;    // Weights for the hidden state (input -> hidden)
    Matrix *recurrent_weights; // Weights for the previous hidden state (hidden -> hidden)
    Matrix *output_weights;    // Weights for the output (hidden -> output)
    Matrix *hidden_bias;       // Bias added before the tanh
    Matrix *output_bias;       // Bias added to the output logits
    Matrix *hidden_state;   // Current hidden state of the RNN
} RNN;

// Initialize and free RNN
RNN *rnn_init(int input_size, int hidden_size, int output_size, double learning_rate);
void rnn_free(RNN *rnn);

// Forward pass: given an input vector, update hidden_state and produce output logits
Matrix *rnn_forward(RNN *rnn, Matrix *input);

// Train on one whole sequence with backpropagation through time.
// inputs[t] is the one-hot word at step t, targets[t] the word that should follow it.
// Returns the summed cross-entropy loss over the sequence.
double rnn_train_sequence(RNN *rnn, Matrix **inputs, Matrix **targets, int length);

// Text generation. temperature <= 0 always picks the most likely word, higher values sample more freely.
char *rnn_generate_text(Vocabulary *v, RNN *rnn, char *initial_input, int length, double temperature);

// Serialize / deserialize
void rnn_save(RNN *rnn, const char *filename);
RNN *rnn_load(const char *filename);
