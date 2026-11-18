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

// sum += m
static void matrix_accumulate(Matrix *sum, Matrix *m)
{
    for (int i = 0; i < sum->rows; i++)
    {
        for (int j = 0; j < sum->cols; j++)
        {
            sum->entries[i][j] += m->entries[i][j];
        }
    }
}

// weights = weights - lr * gradient
static void apply_gradient(Matrix **weights, Matrix *gradient, double learning_rate)
{
    Matrix *scaled_gradient = matrix_scale(learning_rate, gradient);
    Matrix *updated_weights = matrix_subtract(*weights, scaled_gradient);
    matrix_free(scaled_gradient);
    matrix_free(*weights);
    *weights = updated_weights;
}

double rnn_train_sequence(RNN *rnn, Matrix **inputs, Matrix **targets, int length)
{
    if (!rnn || !inputs || !targets || length <= 0)
    {
        fprintf(stderr, "rnn_train_sequence: invalid arguments\n");
        exit(EXIT_FAILURE);
    }

    // Keep every hidden state and output of the sequence for the backward pass.
    // hidden_states[0] is the state before the first word, hidden_states[t + 1] the state after word t.
    Matrix **hidden_states = (Matrix **)malloc((size_t)(length + 1) * sizeof(Matrix *));
    Matrix **outputs = (Matrix **)malloc((size_t)length * sizeof(Matrix *));
    if (!hidden_states || !outputs)
    {
        fprintf(stderr, "rnn_train_sequence: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Forward pass over the whole sequence, starting from a zero hidden state
    matrix_free(rnn->hidden_state);
    rnn->hidden_state = matrix_zero(rnn->hidden_size, 1);
    hidden_states[0] = matrix_copy(rnn->hidden_state);

    double loss = 0.0;
    for (int t = 0; t < length; t++)
    {
        outputs[t] = rnn_forward(rnn, inputs[t]);
        hidden_states[t + 1] = matrix_copy(rnn->hidden_state);
        loss += matrix_cross_entropy(outputs[t], targets[t]);
    }

    // Backward pass through time. Gradients are summed over all steps and
    // applied at the end, so every step uses the weights from the forward pass.
    Matrix *hidden_weights_gradient = matrix_zero(rnn->hidden_size, rnn->input_size);
    Matrix *output_weights_gradient = matrix_zero(rnn->output_size, rnn->hidden_size);
    Matrix *output_weights_transpose = matrix_transpose(rnn->output_weights);
    Matrix *next_hidden_error = matrix_zero(rnn->hidden_size, 1); // error flowing back from step t + 1

    for (int t = length - 1; t >= 0; t--)
    {
        // Softmax + cross-entropy: dL/dlogits = probs - target
        Matrix *output_error = matrix_subtract(outputs[t], targets[t]);

        // dW_o += output_error * h_t^T
        Matrix *hidden_state_transpose = matrix_transpose(hidden_states[t + 1]);
        Matrix *output_step_gradient = matrix_dot(output_error, hidden_state_transpose);
        matrix_accumulate(output_weights_gradient, output_step_gradient);
        matrix_free(hidden_state_transpose);
        matrix_free(output_step_gradient);

        // dL/dh_t = W_o^T * output_error + error from step t + 1
        Matrix *hidden_error = matrix_dot(output_weights_transpose, output_error);
        matrix_accumulate(hidden_error, next_hidden_error);
        matrix_free(output_error);

        // Backprop through tanh: d/dx tanh(x) = 1 - tanh(x)^2, and h_t already holds tanh(x)
        for (int i = 0; i < hidden_error->rows; i++)
        {
            double h = hidden_states[t + 1]->entries[i][0];
            hidden_error->entries[i][0] *= (1.0 - h * h);
        }

        // dW_h += hidden_error * x_t^T
        Matrix *input_transpose = matrix_transpose(inputs[t]);
        Matrix *hidden_step_gradient = matrix_dot(hidden_error, input_transpose);
        matrix_accumulate(hidden_weights_gradient, hidden_step_gradient);
        matrix_free(input_transpose);
        matrix_free(hidden_step_gradient);

        // h_t = tanh(W_h * x_t + h_{t-1}), so the error passes straight on to h_{t-1}
        matrix_free(next_hidden_error);
        next_hidden_error = hidden_error;
    }

    apply_gradient(&rnn->hidden_weights, hidden_weights_gradient, rnn->learning_rate);
    apply_gradient(&rnn->output_weights, output_weights_gradient, rnn->learning_rate);

    matrix_free(hidden_weights_gradient);
    matrix_free(output_weights_gradient);
    matrix_free(output_weights_transpose);
    matrix_free(next_hidden_error);
    for (int t = 0; t < length; t++)
    {
        matrix_free(outputs[t]);
    }
    for (int t = 0; t <= length; t++)
    {
        matrix_free(hidden_states[t]);
    }
    free(outputs);
    free(hidden_states);

    return loss;
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

    // Feed <bos> first, the same way every training sentence starts
    Matrix *bos = create_one_hot_vector(v, "<bos>");
    matrix_free(rnn_forward(rnn, bos));
    matrix_free(bos);

    // Fall back to <unk> for words the model has never seen
    if (vocabulary_get_index(v, initial_input) == -1)
    {
        fprintf(stderr, "Warning: '%s' is not in the vocabulary, using <unk>\n", initial_input);
        initial_input = "<unk>";
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

        // Stop at the end of the sentence
        if (predicted_word_index == VOCAB_EOS)
        {
            matrix_free(output);
            break;
        }

        const char *predicted_word = vocabulary_get_word(v, predicted_word_index);
        if (!predicted_word)
        {
            predicted_word = "<unk>";
        }

        // Append word to buffer
        if (generated_text[0] != '\0')
        {
            strcat(generated_text, " ");
        }
        strncat(generated_text, predicted_word, MAX_WORD_LENGTH);

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
