#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/rnn.h"
#include "vocabulary/vocabulary.h"

// Simple tokenizer that splits on spaces only
static int count_tokens(const char *sentence)
{
    char *copy = strdup(sentence);
    if (!copy)
    {
        fprintf(stderr, "count_tokens: strdup failed\n");
        exit(EXIT_FAILURE);
    }

    int count = 0;
    char *tok = strtok(copy, " ");
    while (tok)
    {
        count++;
        tok = strtok(NULL, " ");
    }

    free(copy);
    return count;
}

int main(void)
{
    // Initialize vocabulary
    Vocabulary *v = vocabulary_create(100);
    if (!v)
    {
        fprintf(stderr, "Failed to create vocabulary\n");
        return EXIT_FAILURE;
    }

    const char *training_data[] = {
        "Matrix dimensions don’t match? Shocking.",
        "Rain on the window? Wow, never seen that before.",
        "Starting is the hardest part? Groundbreaking insight.",
        "Traveling? Because who wouldn’t want to get lost in a new place?",
        "Books? Oh yeah, they’re just full of ideas or whatever.",
        "Time’s too short for pointless stuff... unless it’s procrastination."};

    int num_sentences = (int)(sizeof(training_data) / sizeof(training_data[0]));

    // ----------------------------------------------------------------------
    // 1. Build vocabulary from all tokens in all sentences
    // ----------------------------------------------------------------------
    for (int i = 0; i < num_sentences; i++)
    {
        char *sentence = strdup(training_data[i]);
        if (!sentence)
        {
            fprintf(stderr, "strdup failed while building vocabulary\n");
            vocabulary_free(v);
            return EXIT_FAILURE;
        }

        char *word = strtok(sentence, " ");
        while (word != NULL)
        {
            vocabulary_add_word(v, word);
            word = strtok(NULL, " ");
        }

        free(sentence);
    }

    // ----------------------------------------------------------------------
    // 2. Build all adjacent word pairs across the dataset
    //    For sentence: w0 w1 w2 w3
    //    We create pairs: (<bos>->w0), (w0->w1), (w1->w2), (w2->w3), (w3-><eos>)
    // ----------------------------------------------------------------------

    // First pass: count total pairs (one per token, plus the <eos> pair)
    int total_pairs = 0;
    for (int i = 0; i < num_sentences; i++)
    {
        total_pairs += count_tokens(training_data[i]) + 1;
    }

    if (total_pairs == 0)
    {
        fprintf(stderr, "No training pairs found\n");
        vocabulary_free(v);
        return EXIT_FAILURE;
    }

    Matrix **input_vectors = (Matrix **)malloc((size_t)total_pairs * sizeof(Matrix *));
    Matrix **target_vectors = (Matrix **)malloc((size_t)total_pairs * sizeof(Matrix *));
    int *starts_sentence = (int *)malloc((size_t)total_pairs * sizeof(int)); // 1 if pair is the first of its sentence
    if (!input_vectors || !target_vectors || !starts_sentence)
    {
        fprintf(stderr, "Memory allocation failed for training pairs\n");
        free(input_vectors);
        free(target_vectors);
        free(starts_sentence);
        vocabulary_free(v);
        return EXIT_FAILURE;
    }

    // Second pass: fill in all pairs as one-hot vectors
    int pair_index = 0;
    for (int i = 0; i < num_sentences; i++)
    {
        char *sentence = strdup(training_data[i]);
        if (!sentence)
        {
            fprintf(stderr, "strdup failed while creating training pairs\n");
            // Cleanup partial allocations
            for (int k = 0; k < pair_index; k++)
            {
                matrix_free(input_vectors[k]);
                matrix_free(target_vectors[k]);
            }
            free(input_vectors);
            free(target_vectors);
            free(starts_sentence);
            vocabulary_free(v);
            return EXIT_FAILURE;
        }

        // Every sentence starts with <bos> and its last word predicts <eos>
        char *prev = "<bos>";
        char *curr = strtok(sentence, " ");
        int first_pair = 1;

        while (prev != NULL)
        {
            char *target = curr ? curr : "<eos>";

            input_vectors[pair_index] = create_one_hot_vector(v, prev);
            target_vectors[pair_index] = create_one_hot_vector(v, target);
            starts_sentence[pair_index] = first_pair;
            first_pair = 0;
            pair_index++;

            prev = curr;
            if (curr != NULL)
            {
                curr = strtok(NULL, " ");
            }
        }

        free(sentence);
    }

    if (pair_index != total_pairs)
    {
        fprintf(stderr, "Logic error: pair_index (%d) != total_pairs (%d)\n", pair_index, total_pairs);
        // Cleanup and exit
        for (int k = 0; k < pair_index; k++)
        {
            matrix_free(input_vectors[k]);
            matrix_free(target_vectors[k]);
        }
        free(input_vectors);
        free(target_vectors);
        free(starts_sentence);
        vocabulary_free(v);
        return EXIT_FAILURE;
    }

    // ----------------------------------------------------------------------
    // 3. Initialize RNN
    // ----------------------------------------------------------------------
    int input_size = v->size;
    int hidden_size = 100;
    int output_size = v->size;
    double learning_rate = 0.01;

    RNN *rnn = rnn_init(input_size, hidden_size, output_size, learning_rate);
    if (!rnn)
    {
        fprintf(stderr, "Failed to initialize RNN\n");
        for (int k = 0; k < total_pairs; k++)
        {
            matrix_free(input_vectors[k]);
            matrix_free(target_vectors[k]);
        }
        free(input_vectors);
        free(target_vectors);
        free(starts_sentence);
        vocabulary_free(v);
        return EXIT_FAILURE;
    }

    // ----------------------------------------------------------------------
    // 4. Training loop
    //    Uses the new rnn_backward signature: (rnn, input, target, output)
    //    The hidden state is reset at the start of each sentence and carried
    //    across the words in it, the same way rnn_generate_text uses it.
    // ----------------------------------------------------------------------
    int epochs = 2000;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double epoch_loss = 0.0;

        for (int i = 0; i < total_pairs; i++)
        {
            // Reset hidden state at the start of each sentence (no BPTT)
            if (starts_sentence[i])
            {
                matrix_free(rnn->hidden_state);
                rnn->hidden_state = matrix_zero(rnn->hidden_size, 1);
            }

            Matrix *output = rnn_forward(rnn, input_vectors[i]);
            double loss = matrix_cross_entropy(output, target_vectors[i]);
            epoch_loss += loss;

            // Backward using the already computed output
            rnn_backward(rnn, input_vectors[i], target_vectors[i], output);

            matrix_free(output);
        }

        double avg_epoch_loss = epoch_loss / (double)total_pairs;
        if (epoch % 1000 == 0 || epoch == epochs - 1)
        {
            printf("Epoch %d, Average Loss: %f\n", epoch, avg_epoch_loss);
        }
    }

    // ----------------------------------------------------------------------
    // 5. Generate some text
    // ----------------------------------------------------------------------
    const char *input_text = "Matrix";
    char *next_word_predictions = rnn_generate_text(v, rnn, (char *)input_text, 5);
    printf("Input text: %s\n", input_text);
    printf("Next word predictions: %s\n", next_word_predictions);

    // ----------------------------------------------------------------------
    // 6. Cleanup
    // ----------------------------------------------------------------------
    free(next_word_predictions);

    for (int i = 0; i < total_pairs; i++)
    {
        matrix_free(input_vectors[i]);
        matrix_free(target_vectors[i]);
    }
    free(input_vectors);
    free(target_vectors);
    free(starts_sentence);

    rnn_free(rnn);
    vocabulary_free(v);

    return EXIT_SUCCESS;
}
