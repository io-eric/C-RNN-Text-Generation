#pragma once

#include <stdio.h>

typedef struct
{
    double **entries;
    int rows;
    int cols;
} Matrix;

// Matrix Creation, Management, and Basic Utilities
Matrix *matrix_create(int row, int col);
void matrix_free(Matrix *m);
void matrix_fill(Matrix *m, double n);
Matrix *matrix_zero(int row, int col);
Matrix *matrix_copy(Matrix *m);
void matrix_print(Matrix *m);
void matrix_print_dimensions(Matrix *m);

// File Operations
void matrix_save(Matrix *m, FILE *file);
Matrix *matrix_load(FILE *file);

// Matrix Queries
int matrix_argmax(Matrix *m);
int matrix_check_dimensions(Matrix *m1, Matrix *m2);
Matrix *matrix_row(Matrix *m, int row_index);
double matrix_mean_square_error(Matrix *output, Matrix *target);

// Matrix Operations
void matrix_randomize(Matrix *m, double min, double max);
void matrix_xavier_randomize(Matrix *m, int input_size, int output_size);
double matrix_sum_elements(Matrix *m);
Matrix *matrix_add(Matrix *m1, Matrix *m2);
Matrix *matrix_subtract(Matrix *m1, Matrix *m2);
Matrix *matrix_multiply(Matrix *m1, Matrix *m2);
Matrix *matrix_dot(Matrix *m1, Matrix *m2);
Matrix *matrix_apply(double (*func)(double), Matrix *m);
Matrix *matrix_scale(double n, Matrix *m);
Matrix *matrix_addScalar(double n, Matrix *m);
Matrix *matrix_transpose(Matrix *m);