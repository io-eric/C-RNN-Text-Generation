#pragma once
#include "../matrix/matrix.h"

// Enum for special token IDs
typedef enum
{
    VOCAB_UNK = 0,      // <unk>
    VOCAB_PAD,          // <pad>
    VOCAB_BOS,          // <bos>
    VOCAB_EOS,          // <eos>
    VOCAB_SPECIAL_COUNT // Total number of special tokens
} SpecialTokens;

typedef struct VocabularyEntry
{
    char *word;
    int id;
    struct VocabularyEntry *next; // For chaining (handling collisions)
} VocabularyEntry;

typedef struct
{
    VocabularyEntry **table;
    int size;     // Number of words in the vocabulary
    int capacity; // Capacity of the hash table
} Vocabulary;

Vocabulary *vocabulary_create(int initial_capacity);
void vocabulary_free(Vocabulary *v);
int vocabulary_add_word(Vocabulary *v, const char *word);
int vocabulary_get_index(const Vocabulary *v, const char *word);
char *vocabulary_get_word(Vocabulary *v, int index);
void vocabulary_print(const Vocabulary *v);
Matrix *create_one_hot_vector(Vocabulary *v, char *word);