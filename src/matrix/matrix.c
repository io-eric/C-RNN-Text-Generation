#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

#define MAXCHAR 100

Matrix* matrix_create(int rows, int cols) {
    // Allocate memory for the matrix structure
    Matrix* matrix = malloc(sizeof(Matrix));
    if (!matrix) return NULL;  // Memory allocation failure check

    matrix->rows = rows;
    matrix->cols = cols;

    // Allocate memory for the matrix entries (array of row pointers)
    matrix->entries = malloc(rows * sizeof(double*));
    if (!matrix->entries) {
        free(matrix);  // Free matrix structure if allocation fails
        return NULL;
    }

    // Allocate memory for each row (array of doubles)
    for (int i = 0; i < rows; i++) {
        matrix->entries[i] = malloc(cols * sizeof(double));
        if (!matrix->entries[i]) {
            // Free all previously allocated memory if row allocation fails
            for (int j = 0; j < i; j++) {
                free(matrix->entries[j]);
            }
            free(matrix->entries);
            free(matrix);
            return NULL;
        }
    }

    return matrix;  // Return the created matrix
}

// Function to free the matrix memory
void matrix_free(Matrix* matrix) {
    if (matrix) {
        // Free each row
        for (int i = 0; i < matrix->rows; i++) {
            free(matrix->entries[i]);
        }
        // Free the row pointers array
        free(matrix->entries);
        // Free the matrix structure itself
        free(matrix);
    }
}

void matrix_fill(Matrix *m, double n)
{
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->entries[i][j] = n;
        }
    }
}

Matrix* matrix_zero(int rows, int cols) {
    Matrix* m = matrix_create(rows, cols);
    matrix_fill(m, 0.0);
    return m;
}

Matrix *matrix_copy(Matrix *m)
{
    Matrix *mat = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            mat->entries[i][j] = m->entries[i][j];
        }
    }
    return mat;
}

void matrix_print(Matrix *m)
{
    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            printf("%1.3f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

void matrix_print_dimensions(Matrix *m)
{
    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
}

void matrix_save(Matrix *m, FILE *file)
{
    fprintf(file, "%d\n", m->rows);
    fprintf(file, "%d\n", m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            fprintf(file, "%.6f\n", m->entries[i][j]);
        }
    }
    printf("Successfully saved matrix to file\n");
}

Matrix *matrix_load(FILE *file)
{
    char entry[MAXCHAR];

    // Read matrix dimensions
    if (!fgets(entry, MAXCHAR, file))
        return NULL;
    int rows = atoi(entry);
    if (!fgets(entry, MAXCHAR, file))
        return NULL;
    int cols = atoi(entry);

    // Create matrix
    Matrix *m = matrix_create(rows, cols);
    if (!m)
        return NULL; // Memory allocation failed

    // Read matrix data
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            if (!fgets(entry, MAXCHAR, file))
                return NULL;
            m->entries[i][j] = strtod(entry, NULL);
        }
    }
    return m;
}

int matrix_argmax(Matrix *m)
{
    // Expects a Mx1 matrix
    double max_score = 0;
    int max_idx = 0;
    for (int i = 0; i < m->rows; i++)
    {
        if (m->entries[i][0] > max_score)
        {
            max_score = m->entries[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}

Matrix* matrix_row(Matrix* m, int row_index) {
    if (row_index >= m->rows || row_index < 0) {
        // Invalid row index
        return NULL;
    }

    // Create a new matrix with one row and the same number of columns
    Matrix* row = matrix_create(1, m->cols);

    // Copy the elements of the specified row into the new matrix
    for (int col = 0; col < m->cols; col++) {
        row->entries[0][col] = m->entries[row_index][col];
    }

    return row;
}

double matrix_mean_square_error(Matrix *output, Matrix *target) {
    // Ensure the matrices have the same dimensions
    if (output->rows != target->rows || output->cols != target->cols) {
        fprintf(stderr, "Error: Matrices must have the same dimensions for MSE calculation.\n");
        exit(EXIT_FAILURE); 
    }

    double mse = 0.0;
    int total_elements = output->rows * output->cols;

    // Calculate the sum of squared differences
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            double error = output->entries[i][j] - target->entries[i][j];
            mse += pow(error, 2); // Square the error and add to the total
        }
    }

    // Divide by the total number of elements to get the mean
    mse /= total_elements;

    return mse;
}

int matrix_check_dimensions(Matrix *m1, Matrix *m2)
{
    return m1->rows == m2->rows && m1->cols == m2->cols;
}

void matrix_randomize(Matrix *m, double min, double max)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand((unsigned int)time(NULL));  // Seed random number generator once
        seeded = 1;
    }

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            // Generate random numbers in the range [min, max)
            m->entries[i][j] = min + ((double)rand() / RAND_MAX) * (max - min);
        }
    }
}

void matrix_xavier_randomize(Matrix *m, int input_size, int output_size)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand((unsigned int)time(NULL));  // Seed random number generator once
        seeded = 1;
    }

    double limit = sqrt(6.0 / (input_size + output_size));  // Xavier limit

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            // Initialize weights with values from the uniform distribution within [-limit, limit]
            m->entries[i][j] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * limit;
        }
    }
}

double matrix_sum_elements(Matrix *m)
{
    double sum = 0.0;
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            sum += m->entries[i][j];
        }
    }
    return sum;
}

Matrix *matrix_add(Matrix *m1, Matrix *m2)
{
    if (matrix_check_dimensions(m1, m2))
    {
        Matrix *m = matrix_create(m1->rows, m1->cols);
        for (int i = 0; i < m1->rows; i++)
        {
            for (int j = 0; j < m2->cols; j++)
            {
                m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
            }
        }
        return m;
    }
    else
    {
        printf("(matrix_add) Dimensions mismatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
}
Matrix *matrix_subtract(Matrix *m1, Matrix *m2)
{
    if (matrix_check_dimensions(m1, m2))
    {
        Matrix *m = matrix_create(m1->rows, m1->cols);
        for (int i = 0; i < m1->rows; i++)
        {
            for (int j = 0; j < m2->cols; j++)
            {
                m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
            }
        }
        return m;
    }
    else
    {
        printf("(matrix_subtract) Dimensions mismatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
}

Matrix *matrix_multiply(Matrix *m1, Matrix *m2)
{
    if (m1->cols == m2->rows)
    {
        // Create a result matrix with appropriate dimensions
        Matrix *result = matrix_create(m1->rows, m2->cols);
        
        // Perform matrix multiplication
        for (int i = 0; i < m1->rows; i++)
        {
            for (int j = 0; j < m2->cols; j++)
            {
                result->entries[i][j] = 0;
                for (int k = 0; k < m1->cols; k++)
                {
                    result->entries[i][j] += m1->entries[i][k] * m2->entries[k][j];
                }
            }
        }
        return result;
    }
    else
    {
        printf("(matrix_multiply) Dimensions mismatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
}

Matrix *matrix_dot(Matrix *m1, Matrix *m2)
{
    if (m1->cols == m2->rows)
    {
        Matrix *m = matrix_create(m1->rows, m2->cols);
        for (int i = 0; i < m1->rows; i++)
        {
            for (int j = 0; j < m2->cols; j++)
            {
                double sum = 0;
                for (int k = 0; k < m2->rows; k++)
                {
                    sum += m1->entries[i][k] * m2->entries[k][j];
                }
                m->entries[i][j] = sum;
            }
        }
        return m;
    }
    else
    {
        printf("(matrix_dot) Dimensions mismatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
}

Matrix *matrix_apply(double (*func)(double), Matrix *m)
{
    Matrix *mat = matrix_copy(m);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            mat->entries[i][j] = (*func)(m->entries[i][j]);
        }
    }
    return mat;
}

Matrix *matrix_scale(double n, Matrix *m)
{
    Matrix *mat = matrix_copy(m);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            mat->entries[i][j] *= n;
        }
    }
    return mat;
}

Matrix *matrix_addScalar(double n, Matrix *m)
{
    Matrix *mat = matrix_copy(m);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            mat->entries[i][j] += n;
        }
    }
    return mat;
}

Matrix *matrix_transpose(Matrix *m)
{
    Matrix *mat = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            mat->entries[j][i] = m->entries[i][j];
        }
    }
    return mat;
}
