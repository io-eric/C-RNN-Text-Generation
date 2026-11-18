#pragma once
#include "../matrix/matrix.h"

/**
 * Enum for special token IDs.
 *
 * The insertion order in `vocabulary_create` must match these indices.
 */
typedef enum
{
    VOCAB_PAD = 0,      // <pad>
    VOCAB_BOS,          // <bos>
    VOCAB_EOS,          // <eos>
    VOCAB_UNK,          // <unk>
    VOCAB_SPECIAL_COUNT // Total number of special tokens
} SpecialTokens;

/**
 * Entry in the hash table for the vocabulary.
 * Uses chaining to resolve hash collisions.
 */
typedef struct VocabularyEntry
{
    char *word;
    int id;
    struct VocabularyEntry *next; // For chaining (handling collisions)
} VocabularyEntry;

/**
 * Vocabulary structure.
 *
 * - `table` is a hash table from word -> entry.
 * - `id_to_word` is a direct array mapping id -> word for O(1) lookup.
 *   It is allocated with size at least `capacity`.
 */
typedef struct
{
    VocabularyEntry **table; // hash buckets
    char **id_to_word;       // direct mapping from id -> word
    int size;                // number of words currently in the vocabulary
    int capacity;            // number of buckets / max number of words
} Vocabulary;

Vocabulary *vocabulary_create(int initial_capacity);
void vocabulary_free(Vocabulary *v);

/**
 * Add a word to the vocabulary if it doesn't already exist.
 * Returns the word ID (existing or newly assigned), or -1 on failure.
 */
int vocabulary_add_word(Vocabulary *v, const char *word);

/**
 * Look up the index of a given word.
 * Returns the word ID, or -1 if not found.
 */
int vocabulary_get_index(const Vocabulary *v, const char *word);

/**
 * Look up the word string for a given index.
 * Returns a pointer to the word string, or NULL if index is invalid.
 */
char *vocabulary_get_word(Vocabulary *v, int index);

/**
 * Print all words and their IDs in the vocabulary.
 */
void vocabulary_print(const Vocabulary *v);

/**
 * Split text into tokens. Words are separated by whitespace, and runs of
 * punctuation (. , ? ! ; :) become their own tokens, so "match?" -> "match" "?"
 * and "stuff..." -> "stuff" "...". Apostrophes stay inside words ("don’t").
 * Returns an array of newly allocated strings and stores its length in *count,
 * or NULL on allocation failure. Free with vocabulary_free_tokens.
 */
char **vocabulary_tokenize(const char *text, int *count);
void vocabulary_free_tokens(char **tokens, int count);

/**
 * Returns 1 if the token is punctuation produced by vocabulary_tokenize.
 */
int vocabulary_is_punctuation(const char *token);

/**
 * Create a one-hot encoded vector of size (v->size x 1) for the given word.
 * Exits the program if the word is not found.
 */
Matrix *create_one_hot_vector(Vocabulary *v, char *word);