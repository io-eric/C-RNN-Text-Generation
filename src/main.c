#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model/rnn.h"
#include "vocabulary/vocabulary.h"

int main()
{
    // Initialize vocabulary
    Vocabulary *v = vocabulary_create(100);

    const char *training_data[] = {
        "Matrix dimensions don’t match? Shocking.",
        "Rain on the window? Wow, never seen that before.",
        "Starting is the hardest part? Groundbreaking insight.",
        "Traveling? Because who wouldn’t want to get lost in a new place?",
        "Books? Oh yeah, they’re just full of ideas or whatever.",
        "Time’s too short for pointless stuff... unless it’s procrastination."};

    int num_samples = sizeof(training_data) / sizeof(training_data[0]);

    // Add words to the vocabulary
    for (int i = 0; i < num_samples; i++)
    {
        char *sentence = strdup(training_data[i]);
        char *word = strtok(sentence, " ");
        while (word != NULL)
        {
            vocabulary_add_word(v, word);
            word = strtok(NULL, " ");
        }
        free(sentence);
    }

    // Parameters
    int input_size = v->size;
    int hidden_size = 100;
    int output_size = v->size;
    double learning_rate = 0.0001;

    RNN *rnn = rnn_init(input_size, hidden_size, output_size, learning_rate);

    // Prepare training data: input-target pairs
    Matrix **input_vectors = (Matrix **)malloc(num_samples * sizeof(Matrix *));
    Matrix **target_vectors = (Matrix **)malloc(num_samples * sizeof(Matrix *));

    if (input_vectors == NULL || target_vectors == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Convert sentences to sequences of one-hot encoded vectors
    for (int i = 0; i < num_samples; i++)
    {
        char *sentence = strdup(training_data[i]);
        char *word = strtok(sentence, " ");
        input_vectors[i] = create_one_hot_vector(v, word);

        word = strtok(NULL, " ");
        target_vectors[i] = create_one_hot_vector(v, word);

        free(sentence);
    }

    // Train the RNN
    int epochs = 5000;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double epoch_loss = 0.0;
        for (int i = 0; i < num_samples; i++)
        {
            Matrix *output = rnn_forward(rnn, input_vectors[i]);
            rnn_backward(rnn, input_vectors[i], target_vectors[i]);

            double loss = matrix_mean_square_error(output, target_vectors[i]);
            epoch_loss += loss;

            matrix_free(output);
        }

        double avg_epoch_loss = epoch_loss / num_samples;
        if (epoch % 1000 == 0) // Print loss every 1000 epochs
        {
            printf("Epoch %d, Average Loss: %f\n", epoch, avg_epoch_loss);
        }
    }

    // Generate text after training
    char *input_text = "Matrix";
    char *next_word_predictions = rnn_generate_text(v, rnn, input_text, 5);
    printf("Input text: %s\n", input_text);
    printf("Next word predictions: %s\n", next_word_predictions);

    // Clean up
    free(next_word_predictions);
    for (int i = 0; i < num_samples; i++)
    {
        matrix_free(input_vectors[i]);
        matrix_free(target_vectors[i]);
    }
    free(input_vectors);
    free(target_vectors);
    rnn_free(rnn);
    vocabulary_free(v);

    return 0;
}