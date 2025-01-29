#include "vocabulary.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned int hash_word(const char *word)
{
    unsigned int hash = 0;
    while (*word)
    {
        hash = (hash * 31) + *word++;
    }
    return hash;
}

Vocabulary *vocabulary_create(int initial_capacity)
{
    Vocabulary *v = (Vocabulary *)malloc(sizeof(Vocabulary));
    if (!v)
        return NULL; // Memory allocation failure

    initial_capacity += VOCAB_SPECIAL_COUNT;

    v->table = (VocabularyEntry **)calloc(initial_capacity, sizeof(VocabularyEntry *));
    if (!v->table)
    {
        perror("Failed to allocate memory for hash table");
        free(v);
        return NULL; // Memory allocation failure
    }
    v->size = 0;
    v->capacity = initial_capacity;

    // Add special tokens with predefined IDs
    vocabulary_add_word(v, "<pad>"); // ID 0
    vocabulary_add_word(v, "<bos>"); // ID 1
    vocabulary_add_word(v, "<eos>"); // ID 2
    vocabulary_add_word(v, "<unk>"); // ID 3

    return v;
}

void vocabulary_free(Vocabulary *v)
{
    if (!v)
        return;
    for (int i = 0; i < v->capacity; i++)
    {
        VocabularyEntry *entry = v->table[i];
        while (entry)
        {
            VocabularyEntry *next = entry->next;
            free(entry->word);
            free(entry);
            entry = next;
        }
    }
    free(v->table);
    free(v);
}

int vocabulary_add_word(Vocabulary *v, const char *word)
{
    if (v->size == v->capacity)
    {
        perror("Vocabulary already full");
        return -1; // Vocabulary full
    }
    unsigned int hash = hash_word(word) % v->capacity;
    VocabularyEntry *entry = v->table[hash];
    while (entry)
    {
        if (strcmp(entry->word, word) == 0)
        {
            return entry->id;
        }
        entry = entry->next;
    }

    VocabularyEntry *new_entry = (VocabularyEntry *)malloc(sizeof(VocabularyEntry));
    if (!new_entry)
    {
        perror("Failed to allocate memory for new entry");
        return -1; // Memory allocation failure
    }

    new_entry->word = strdup(word);
    if (!new_entry->word)
    {
        perror("Failed to allocate memory for word");
        free(new_entry); // Free the entry structure
        return -1;
    }

    new_entry->id = v->size;
    new_entry->next = v->table[hash];
    v->table[hash] = new_entry;

    v->size++;
    return new_entry->id;
}

int vocabulary_lookup_id(const Vocabulary *v, const char *word)
{
    if (!v || !word)
        return -1;

    unsigned int hash = hash_word(word) % v->capacity;
    VocabularyEntry *entry = v->table[hash];

    while (entry)
    {
        if (strcmp(entry->word, word) == 0)
        {
            return entry->id;
        }
        entry = entry->next;
    }
    return -1; // Word not found
}

char *vocabulary_get_word(Vocabulary *v, int index) {
    for (int i = 0; i < v->capacity; i++) {
        VocabularyEntry *entry = v->table[i];
        while (entry) {
            if (entry->id == index) {
                return entry->word;
            }
            entry = entry->next;
        }
    }
    return NULL; // Word not found
}


void vocabulary_print(const Vocabulary *v)
{
    if (!v)
    {
        fprintf(stderr, "Vocabulary is NULL\n");
        return;
    }

    for (int i = 0; i < v->capacity; i++)
    {
        VocabularyEntry *entry = v->table[i];
        while (entry)
        {
            printf("Word: %s, ID: %d\n", entry->word, entry->id);
            entry = entry->next;
        }
    }
}

// Function to create a one-hot encoded vector for a given word
Matrix *create_one_hot_vector(Vocabulary *v, char *word)
{
    int index = vocabulary_lookup_id(v, word);
    if (index == -1)
    {
        fprintf(stderr, "Word '%s' not found in vocabulary\n", word);
        exit(1);
    }
    Matrix *one_hot = matrix_create(v->size, 1);
    for (int i = 0; i < v->size; i++)
    {
        one_hot->entries[i][0] = (i == index) ? 1.0 : 0.0;
    }
    return one_hot;
}