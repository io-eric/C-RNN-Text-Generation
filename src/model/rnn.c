#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

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
    matrix_xavier_randomize(rnn->hidden_weights, input_size, hidden_size);

    // Initialize output weights (hidden to output)
    rnn->output_weights = matrix_create(output_size, hidden_size);
    matrix_xavier_randomize(rnn->output_weights, hidden_size, output_size);

    // Initialize hidden state to zeros
    rnn->hidden_state = matrix_zero(hidden_size, 1);

    return rnn;
}

void rnn_free(RNN *rnn)
{
    if (rnn)
    {
        matrix_free(rnn->hidden_weights);
        matrix_free(rnn->output_weights);
        matrix_free(rnn->hidden_state);
        free(rnn);
    }
}

Matrix *rnn_forward(RNN *rnn, Matrix *input)
{
    // Update hidden state: hidden_state = tanh(hidden_weigths * input + hidden_state)
    Matrix *hidden_input = matrix_dot(rnn->hidden_weights, input);
    Matrix *new_hidden_state = matrix_add(hidden_input, rnn->hidden_state);
    matrix_free(hidden_input);

    // Apply tanh activation to the hidden state
    Matrix *activated_hidden = matrix_apply(tanh, new_hidden_state);
    matrix_free(new_hidden_state);

    // Update the RNN's hidden state
    matrix_free(rnn->hidden_state);
    rnn->hidden_state = matrix_copy(activated_hidden);

    // Compute output: output = output_weights * hidden_state
    Matrix *output = matrix_dot(rnn->output_weights, activated_hidden);
    matrix_free(activated_hidden);

    return output;
}

double square(double x)
{
    return x * x;
}

void rnn_backward(RNN *rnn, Matrix *input, Matrix *target)
{
    // Perform forward pass to get the output and hidden state
    Matrix *output = rnn_forward(rnn, input);

    // Compute the error in the output layer: output_error = output - target
    Matrix *output_error = matrix_subtract(output, target);

    // Compute the gradient of the loss with respect to the output weights
    // output_weights_gradient = output_error * hidden_state^T
    Matrix *hidden_state_transpose = matrix_transpose(rnn->hidden_state);
    Matrix *output_weights_gradient = matrix_dot(output_error, hidden_state_transpose);
    matrix_free(hidden_state_transpose);

    // Update the output weights: output_weights -= learning_rate * output_weights_gradient
    Matrix *scaled_output_weights_gradient = matrix_scale(rnn->learning_rate, output_weights_gradient);
    Matrix *updated_output_weights = matrix_subtract(rnn->output_weights, scaled_output_weights_gradient);
    matrix_free(rnn->output_weights);
    rnn->output_weights = updated_output_weights;
    matrix_free(scaled_output_weights_gradient);
    matrix_free(output_weights_gradient);

    // Compute the gradient of the loss with respect to the hidden state
    // hidden_error = output_weights^T * output_error
    Matrix *output_weights_transpose = matrix_transpose(rnn->output_weights);
    Matrix *hidden_error = matrix_dot(output_weights_transpose, output_error);
    matrix_free(output_weights_transpose);

    // Compute the gradient of the loss with respect to the hidden weights
    // hidden_weights_gradient = hidden_error * input^T
    Matrix *input_transpose = matrix_transpose(input);
    Matrix *hidden_weights_gradient = matrix_dot(hidden_error, input_transpose);
    matrix_free(input_transpose);

    // Update the hidden weights: hidden_weights -= learning_rate * hidden_weights_gradient
    Matrix *scaled_hidden_weights_gradient = matrix_scale(rnn->learning_rate, hidden_weights_gradient);
    Matrix *updated_hidden_weights = matrix_subtract(rnn->hidden_weights, scaled_hidden_weights_gradient);
    matrix_free(rnn->hidden_weights);
    rnn->hidden_weights = updated_hidden_weights;
    matrix_free(scaled_hidden_weights_gradient);
    matrix_free(hidden_weights_gradient);

    // Update the hidden state for the next iteration
    matrix_free(rnn->hidden_state);
    rnn->hidden_state = matrix_copy(hidden_error);

    // Free memory
    matrix_free(output);
    matrix_free(output_error);
    matrix_free(hidden_error);
}

// Generate text using the RNN
char *rnn_generate_text(Vocabulary* v, RNN *rnn, char *initial_input, int length)
{
    // Convert initial_input to a matrix (one-hot encoded or otherwise)
    Matrix *input = matrix_create(rnn->input_size, 1);

    // Initialize the generated text buffer
    // Allocate enough space for the generated words and spaces between them
    char *generated_text = (char *)malloc((length * (MAX_WORD_LENGTH + 1)) * sizeof(char));
    if (!generated_text)
    {
        fprintf(stderr, "Error: Unable to allocate memory for generated text\n");
        exit(1);
    }
    generated_text[0] = '\0'; // Initialize as an empty string

    // Convert the initial input into a one-hot encoded vector and set it as the input
    Matrix *initial_input_vector = create_one_hot_vector(v, initial_input);
    Matrix *input_copy = matrix_copy(initial_input_vector); 
    matrix_free(initial_input_vector); 

    // Copy the input to the matrix input
    for (int i = 0; i < rnn->input_size; i++) {
        input->entries[i][0] = input_copy->entries[i][0];
    }
    matrix_free(input_copy);

    // Generate text
    for (int i = 0; i < length; i++)
    {
        // Perform forward pass
        Matrix *output = rnn_forward(rnn, input);

        // Find the index of the word with the highest probability
        int predicted_word_index = matrix_argmax(output);

        // Map the predicted index to the actual word in the vocabulary
        const char *predicted_word = vocabulary_get_word(v, predicted_word_index);

        // Append the predicted word to the generated text
        strcat(generated_text, predicted_word);
        if (i < length - 1) {
            strcat(generated_text, " "); // Add a space between words
        }

        // Free the output matrix
        matrix_free(output);

        // Update input for the next step
        matrix_fill(input, 0.0);
        input->entries[predicted_word_index][0] = 1.0; // One-hot encoding of predicted word
    }

    // Free the input matrix
    matrix_free(input);

    return generated_text;
}

// Save the RNN model to a file
void rnn_save(RNN *rnn, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        fprintf(stderr, "Error: Unable to open file for saving RNN\n");
        exit(1);
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

// Load the RNN model from a file
RNN *rnn_load(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Error: Unable to open file for loading RNN\n");
        exit(EXIT_FAILURE);
    }

    // Load RNN metadata
    int input_size, hidden_size, output_size;
    double learning_rate;
    fread(&input_size, sizeof(int), 1, file);
    fread(&hidden_size, sizeof(int), 1, file);
    fread(&output_size, sizeof(int), 1, file);
    fread(&learning_rate, sizeof(double), 1, file);

    // Initialize RNN
    RNN *rnn = rnn_init(input_size, hidden_size, output_size, learning_rate);

    // Load matrices
    matrix_free(rnn->hidden_weights);
    matrix_free(rnn->output_weights);
    matrix_free(rnn->hidden_state);
    rnn->hidden_weights = matrix_load(file);
    rnn->output_weights = matrix_load(file);
    rnn->hidden_state = matrix_load(file);

    fclose(file);
    return rnn;
}