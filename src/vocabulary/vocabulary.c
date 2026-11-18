#include "vocabulary.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Simple string hash for vocabulary words */
static unsigned int hash_word(const char *word)
{
    unsigned int hash = 0;
    while (*word)
    {
        hash = (hash * 31u) + (unsigned char)(*word++);
    }
    return hash;
}

Vocabulary *vocabulary_create(int initial_capacity)
{
    Vocabulary *v = (Vocabulary *)malloc(sizeof(Vocabulary));
    if (!v)
    {
        perror("Failed to allocate Vocabulary");
        return NULL;
    }

    /* Reserve space for special tokens as well */
    initial_capacity += VOCAB_SPECIAL_COUNT;
    if (initial_capacity <= 0)
    {
        initial_capacity = VOCAB_SPECIAL_COUNT;
    }

    v->table = (VocabularyEntry **)calloc((size_t)initial_capacity,
                                          sizeof(VocabularyEntry *));
    if (!v->table)
    {
        perror("Failed to allocate memory for hash table");
        free(v);
        return NULL;
    }

    v->id_to_word = (char **)calloc((size_t)initial_capacity, sizeof(char *));
    if (!v->id_to_word)
    {
        perror("Failed to allocate memory for id_to_word");
        free(v->table);
        free(v);
        return NULL;
    }

    v->size = 0;
    v->capacity = initial_capacity;

    /* Add special tokens with predefined IDs matching the enum */
    if (vocabulary_add_word(v, "<pad>") != VOCAB_PAD ||
        vocabulary_add_word(v, "<bos>") != VOCAB_BOS ||
        vocabulary_add_word(v, "<eos>") != VOCAB_EOS ||
        vocabulary_add_word(v, "<unk>") != VOCAB_UNK)
    {
        fprintf(stderr, "Special token IDs do not match the enum\n");
        vocabulary_free(v);
        return NULL;
    }

    return v;
}

void vocabulary_free(Vocabulary *v)
{
    if (!v)
        return;

    /* Free chained hash-table entries and strings */
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
    free(v->id_to_word);
    free(v);
}

int vocabulary_add_word(Vocabulary *v, const char *word)
{
    if (!v || !word)
        return -1;

    unsigned int hash = hash_word(word) % (unsigned int)v->capacity;
    VocabularyEntry *entry = v->table[hash];

    /* Check if word already exists */
    while (entry)
    {
        if (strcmp(entry->word, word) == 0)
        {
            return entry->id;
        }
        entry = entry->next;
    }

    if (v->size >= v->capacity)
    {
        /* For now, we don't implement dynamic resize; fail if full */
        fprintf(stderr, "Vocabulary already full\n");
        return -1;
    }

    /* Create new entry */
    VocabularyEntry *new_entry = (VocabularyEntry *)malloc(sizeof(VocabularyEntry));
    if (!new_entry)
    {
        perror("Failed to allocate memory for new VocabularyEntry");
        return -1;
    }

    new_entry->word = strdup(word);
    if (!new_entry->word)
    {
        perror("Failed to allocate memory for word string");
        free(new_entry);
        return -1;
    }

    new_entry->id = v->size;
    new_entry->next = v->table[hash];
    v->table[hash] = new_entry;

    /* Update id_to_word for O(1) index lookup */
    v->id_to_word[new_entry->id] = new_entry->word;

    v->size++;
    return new_entry->id;
}

int vocabulary_get_index(const Vocabulary *v, const char *word)
{
    if (!v || !word)
        return -1;

    unsigned int hash = hash_word(word) % (unsigned int)v->capacity;
    VocabularyEntry *entry = v->table[hash];

    while (entry)
    {
        if (strcmp(entry->word, word) == 0)
        {
            return entry->id;
        }
        entry = entry->next;
    }

    return -1; /* Word not found */
}

char *vocabulary_get_word(Vocabulary *v, int index)
{
    if (!v || index < 0 || index >= v->size)
        return NULL;

    return v->id_to_word[index];
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

static int is_punctuation_char(char c)
{
    return c != '\0' && strchr(".,?!;:", c) != NULL;
}

int vocabulary_is_punctuation(const char *token)
{
    return token != NULL && is_punctuation_char(token[0]);
}

char **vocabulary_tokenize(const char *text, int *count)
{
    *count = 0;
    if (!text)
        return NULL;

    /* Every token is at least one character long, so strlen + 1 slots is always enough */
    char **tokens = (char **)malloc((strlen(text) + 1) * sizeof(char *));
    if (!tokens)
    {
        perror("Failed to allocate memory for tokens");
        return NULL;
    }

    const char *p = text;
    while (*p)
    {
        if (isspace((unsigned char)*p))
        {
            p++;
            continue;
        }

        /* A token is either a run of punctuation or a run of anything else up to a space or punctuation */
        const char *start = p;
        if (is_punctuation_char(*p))
        {
            while (is_punctuation_char(*p))
                p++;
        }
        else
        {
            while (*p && !isspace((unsigned char)*p) && !is_punctuation_char(*p))
                p++;
        }

        size_t length = (size_t)(p - start);
        char *token = (char *)malloc(length + 1);
        if (!token)
        {
            perror("Failed to allocate memory for token");
            vocabulary_free_tokens(tokens, *count);
            *count = 0;
            return NULL;
        }
        memcpy(token, start, length);
        token[length] = '\0';
        tokens[(*count)++] = token;
    }

    return tokens;
}

void vocabulary_free_tokens(char **tokens, int count)
{
    if (!tokens)
        return;
    for (int i = 0; i < count; i++)
    {
        free(tokens[i]);
    }
    free(tokens);
}

/* Create a one-hot encoded column vector for a given word.
   Exits the program if the word is not in the vocabulary. */
Matrix *create_one_hot_vector(Vocabulary *v, char *word)
{
    if (!v || !word)
    {
        fprintf(stderr, "create_one_hot_vector: invalid arguments\n");
        exit(EXIT_FAILURE);
    }

    int index = vocabulary_get_index(v, word);
    if (index == -1)
    {
        fprintf(stderr, "Word '%s' not found in vocabulary\n", word);
        exit(EXIT_FAILURE);
    }

    Matrix *one_hot = matrix_create(v->size, 1);
    if (!one_hot)
    {
        fprintf(stderr, "create_one_hot_vector: failed to allocate matrix\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < v->size; i++)
    {
        one_hot->entries[i][0] = (i == index) ? 1.0 : 0.0;
    }

    return one_hot;
}