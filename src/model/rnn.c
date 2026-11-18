#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "rnn.h"
#include "../matrix/matrix.h"
#include "../vocabulary/vocabulary.h"

#define MAX_WORD_LENGTH 64

RNN *rnn_init(int input_size, int hidden_size, int output_size, double learning_rate)
{
    RNN *rnn = (RNN *)malloc(sizeof(RNN));
    if (!rnn)
    {
        fprintf(stderr, "Error: Unable to allocate memory for RNN\n");
        exit(EXIT_FAILURE);
    }

    rnn->input_size = input_size;
    rnn->hidden_size = hidden_size;
    rnn->output_size = output_size;
    rnn->learning_rate = learning_rate;

    // Initialize hidden weights (input to hidden)
    rnn->hidden_weights = matrix_create(hidden_size, input_size);
    if (!rnn->hidden_weights)
    {
        fprintf(stderr, "Error: Unable to allocate memory for hidden_weights\n");
        free(rnn);
        exit(EXIT_FAILURE);
    }
    matrix_xavier_randomize(rnn->hidden_weights, input_size, hidden_size);

    // Initialize output weights (hidden to output)
    rnn->output_weights = matrix_create(output_size, hidden_size);
    if (!rnn->output_weights)
    {
        fprintf(stderr, "Error: Unable to allocate memory for output_weights\n");
        matrix_free(rnn->hidden_weights);
        free(rnn);
        exit(EXIT_FAILURE);
    }
    matrix_xavier_randomize(rnn->output_weights, hidden_size, output_size);

    // Initialize hidden state to zeros
    rnn->hidden_state = matrix_zero(hidden_size, 1);
    if (!rnn->hidden_state)
    {
        fprintf(stderr, "Error: Unable to allocate memory for hidden_state\n");
        matrix_free(rnn->hidden_weights);
        matrix_free(rnn->output_weights);
        free(rnn);
        exit(EXIT_FAILURE);
    }

    return rnn;
}

void rnn_free(RNN *rnn)
{
    if (rnn)
    {
        if (rnn->hidden_weights)
            matrix_free(rnn->hidden_weights);
        if (rnn->output_weights)
            matrix_free(rnn->output_weights);
        if (rnn->hidden_state)
            matrix_free(rnn->hidden_state);
        free(rnn);
    }
}

Matrix *rnn_forward(RNN *rnn, Matrix *input)
{
    if (!rnn || !input)
    {
        fprintf(stderr, "rnn_forward: NULL argument\n");
        exit(EXIT_FAILURE);
    }

    // hidden_state = tanh(W_h * input + hidden_state)
    Matrix *hidden_input = matrix_dot(rnn->hidden_weights, input);
    if (!hidden_input)
    {
        fprintf(stderr, "rnn_forward: hidden_input allocation failed\n");
        exit(EXIT_FAILURE);
    }

    Matrix *new_hidden_state = matrix_add(hidden_input, rnn->hidden_state);
    matrix_free(hidden_input);
    if (!new_hidden_state)
    {
        fprintf(stderr, "rnn_forward: new_hidden_state allocation failed\n");
        exit(EXIT_FAILURE);
    }

    Matrix *activated_hidden = matrix_apply(tanh, new_hidden_state);
    matrix_free(new_hidden_state);
    if (!activated_hidden)
    {
        fprintf(stderr, "rnn_forward: activated_hidden allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Update the RNN's hidden state
    matrix_free(rnn->hidden_state);
    rnn->hidden_state = matrix_copy(activated_hidden);
    if (!rnn->hidden_state)
    {
        fprintf(stderr, "rnn_forward: hidden_state copy failed\n");
        matrix_free(activated_hidden);
        exit(EXIT_FAILURE);
    }

    // Compute output logits: logits = W_o * hidden_state
    Matrix *logits = matrix_dot(rnn->output_weights, activated_hidden);
    matrix_free(activated_hidden);
    if (!logits)
    {
        fprintf(stderr, "rnn_forward: logits allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Convert logits to probabilities with softmax for use with cross-entropy
    Matrix *probs = matrix_softmax(logits);
    matrix_free(logits);
    if (!probs)
    {
        fprintf(stderr, "rnn_forward: softmax allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // NOTE: Caller now receives probabilities, not raw logits
    return probs;
}

void rnn_backward(RNN *rnn, Matrix *input, Matrix *target, Matrix *output)
{
    if (!rnn || !input || !target || !output)
    {
        fprintf(stderr, "rnn_backward: NULL argument\n");
        exit(EXIT_FAILURE);
    }

    // Compute the error in the output layer for softmax + cross-entropy:
    // output is expected to be probabilities; with one-hot targets,
    // dL/dlogits simplifies to (probs - target)
    Matrix *output_error = matrix_subtract(output, target);
    if (!output_error)
    {
        fprintf(stderr, "rnn_backward: output_error allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Hidden error: hidden_error = W_o^T * output_error
    // Computed before W_o is updated so it uses the same weights as the forward pass
    Matrix *output_weights_transpose = matrix_transpose(rnn->output_weights);
    if (!output_weights_transpose)
    {
        fprintf(stderr, "rnn_backward: output_weights_transpose allocation failed\n");
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    Matrix *hidden_error = matrix_dot(output_weights_transpose, output_error);
    matrix_free(output_weights_transpose);
    if (!hidden_error)
    {
        fprintf(stderr, "rnn_backward: hidden_error allocation failed\n");
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    // Gradient for output weights: dW_o = output_error * hidden_state^T
    Matrix *hidden_state_transpose = matrix_transpose(rnn->hidden_state);
    if (!hidden_state_transpose)
    {
        fprintf(stderr, "rnn_backward: hidden_state_transpose allocation failed\n");
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    Matrix *output_weights_gradient = matrix_dot(output_error, hidden_state_transpose);
    matrix_free(hidden_state_transpose);
    if (!output_weights_gradient)
    {
        fprintf(stderr, "rnn_backward: output_weights_gradient allocation failed\n");
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    // Update output weights: W_o = W_o - lr * dW_o
    Matrix *scaled_output_weights_gradient = matrix_scale(rnn->learning_rate, output_weights_gradient);
    if (!scaled_output_weights_gradient)
    {
        fprintf(stderr, "rnn_backward: scaled_output_weights_gradient allocation failed\n");
        matrix_free(output_weights_gradient);
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    Matrix *updated_output_weights = matrix_subtract(rnn->output_weights, scaled_output_weights_gradient);
    if (!updated_output_weights)
    {
        fprintf(stderr, "rnn_backward: updated_output_weights allocation failed\n");
        matrix_free(scaled_output_weights_gradient);
        matrix_free(output_weights_gradient);
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    matrix_free(rnn->output_weights);
    rnn->output_weights = updated_output_weights;
    matrix_free(scaled_output_weights_gradient);
    matrix_free(output_weights_gradient);

    // Backprop through tanh: d/dx tanh(x) = 1 - tanh(x)^2, and hidden_state already holds tanh(x)
    for (int i = 0; i < hidden_error->rows; i++)
    {
        double h = rnn->hidden_state->entries[i][0];
        hidden_error->entries[i][0] *= (1.0 - h * h);
    }

    // Gradient for hidden weights: dW_h = hidden_error * input^T
    Matrix *input_transpose = matrix_transpose(input);
    if (!input_transpose)
    {
        fprintf(stderr, "rnn_backward: input_transpose allocation failed\n");
        matrix_free(hidden_error);
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    Matrix *hidden_weights_gradient = matrix_dot(hidden_error, input_transpose);
    matrix_free(input_transpose);
    if (!hidden_weights_gradient)
    {
        fprintf(stderr, "rnn_backward: hidden_weights_gradient allocation failed\n");
        matrix_free(hidden_error);
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    // Update hidden weights: W_h = W_h - lr * dW_h
    Matrix *scaled_hidden_weights_gradient = matrix_scale(rnn->learning_rate, hidden_weights_gradient);
    if (!scaled_hidden_weights_gradient)
    {
        fprintf(stderr, "rnn_backward: scaled_hidden_weights_gradient allocation failed\n");
        matrix_free(hidden_weights_gradient);
        matrix_free(hidden_error);
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    Matrix *updated_hidden_weights = matrix_subtract(rnn->hidden_weights, scaled_hidden_weights_gradient);
    if (!updated_hidden_weights)
    {
        fprintf(stderr, "rnn_backward: updated_hidden_weights allocation failed\n");
        matrix_free(scaled_hidden_weights_gradient);
        matrix_free(hidden_weights_gradient);
        matrix_free(hidden_error);
        matrix_free(output_error);
        exit(EXIT_FAILURE);
    }

    matrix_free(rnn->hidden_weights);
    rnn->hidden_weights = updated_hidden_weights;
    matrix_free(scaled_hidden_weights_gradient);
    matrix_free(hidden_weights_gradient);

    // For this simple one-step training, we don't propagate hidden_error back in time.
    // Caller can choose when to reset rnn->hidden_state (e.g., between sequences).

    matrix_free(output_error);
    matrix_free(hidden_error);
}

char *rnn_generate_text(Vocabulary *v, RNN *rnn, char *initial_input, int length)
{
    if (!v || !rnn || !initial_input || length <= 0)
    {
        fprintf(stderr, "rnn_generate_text: invalid arguments\n");
        exit(EXIT_FAILURE);
    }

    // Reset hidden state before generation
    if (rnn->hidden_state)
        matrix_free(rnn->hidden_state);
    rnn->hidden_state = matrix_zero(rnn->hidden_size, 1);
    if (!rnn->hidden_state)
    {
        fprintf(stderr, "rnn_generate_text: hidden_state allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Create one-hot of initial input
    Matrix *input = create_one_hot_vector(v, initial_input);
    if (!input)
    {
        fprintf(stderr, "rnn_generate_text: failed to create initial input vector\n");
        exit(EXIT_FAILURE);
    }

    // Allocate enough space for generated words and spaces
    char *generated_text = (char *)malloc((length * (MAX_WORD_LENGTH + 1)) * sizeof(char));
    if (!generated_text)
    {
        fprintf(stderr, "rnn_generate_text: Unable to allocate memory for generated text\n");
        matrix_free(input);
        exit(EXIT_FAILURE);
    }
    generated_text[0] = '\0';

    for (int i = 0; i < length; i++)
    {
        Matrix *output = rnn_forward(rnn, input);
        if (!output)
        {
            fprintf(stderr, "rnn_generate_text: forward pass failed\n");
            matrix_free(input);
            free(generated_text);
            exit(EXIT_FAILURE);
        }

        int predicted_word_index = matrix_argmax(output);
        const char *predicted_word = vocabulary_get_word(v, predicted_word_index);
        if (!predicted_word)
        {
            predicted_word = "<unk>";
        }

        // Append word to buffer
        strncat(generated_text, predicted_word, MAX_WORD_LENGTH);
        if (i < length - 1)
        {
            strcat(generated_text, " ");
        }

        // Prepare next input as one-hot of predicted word
        matrix_fill(input, 0.0);
        if (predicted_word_index >= 0 && predicted_word_index < input->rows)
        {
            input->entries[predicted_word_index][0] = 1.0;
        }

        matrix_free(output);
    }

    matrix_free(input);
    return generated_text;
}

void rnn_save(RNN *rnn, const char *filename)
{
    if (!rnn || !filename)
    {
        fprintf(stderr, "rnn_save: invalid arguments\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        fprintf(stderr, "Error: Unable to open file for saving RNN: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Save RNN metadata
    fwrite(&rnn->input_size, sizeof(int), 1, file);
    fwrite(&rnn->hidden_size, sizeof(int), 1, file);
    fwrite(&rnn->output_size, sizeof(int), 1, file);
    fwrite(&rnn->learning_rate, sizeof(double), 1, file);

    // Save matrices
    matrix_save(rnn->hidden_weights, file);
    matrix_save(rnn->output_weights, file);
    matrix_save(rnn->hidden_state, file);

    fclose(file);
}

RNN *rnn_load(const char *filename)
{
    if (!filename)
    {
        fprintf(stderr, "rnn_load: filename is NULL\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Error: Unable to open file for loading RNN: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int input_size, hidden_size, output_size;
    double learning_rate;

    if (fread(&input_size, sizeof(int), 1, file) != 1 ||
        fread(&hidden_size, sizeof(int), 1, file) != 1 ||
        fread(&output_size, sizeof(int), 1, file) != 1 ||
        fread(&learning_rate, sizeof(double), 1, file) != 1)
    {
        fprintf(stderr, "rnn_load: failed to read RNN metadata\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    RNN *rnn = rnn_init(input_size, hidden_size, output_size, learning_rate);
    if (!rnn)
    {
        fprintf(stderr, "rnn_load: rnn_init failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Replace matrices with those loaded from file
    Matrix *loaded_hidden_weights = matrix_load(file);
    Matrix *loaded_output_weights = matrix_load(file);
    Matrix *loaded_hidden_state = matrix_load(file);

    if (!loaded_hidden_weights || !loaded_output_weights || !loaded_hidden_state)
    {
        fprintf(stderr, "rnn_load: failed to load one or more matrices\n");
        if (loaded_hidden_weights) matrix_free(loaded_hidden_weights);
        if (loaded_output_weights) matrix_free(loaded_output_weights);
        if (loaded_hidden_state) matrix_free(loaded_hidden_state);
        rnn_free(rnn);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    matrix_free(rnn->hidden_weights);
    matrix_free(rnn->output_weights);
    matrix_free(rnn->hidden_state);

    rnn->hidden_weights = loaded_hidden_weights;
    rnn->output_weights = loaded_output_weights;
    rnn->hidden_state = loaded_hidden_state;

    fclose(file);
    return rnn;
}
