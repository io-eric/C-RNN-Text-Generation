#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "matrix.h"

#define MAXCHAR 100

Matrix *matrix_create(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        fprintf(stderr, "matrix_create: invalid dimensions %d x %d\n", rows, cols);
        return NULL;
    }

    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    if (!matrix)
    {
        fprintf(stderr, "matrix_create: failed to allocate Matrix struct\n");
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->entries = (double **)malloc(rows * sizeof(double *));
    if (!matrix->entries)
    {
        fprintf(stderr, "matrix_create: failed to allocate row pointers\n");
        free(matrix);
        return NULL;
    }

    for (int i = 0; i < rows; i++)
    {
        matrix->entries[i] = (double *)malloc(cols * sizeof(double));
        if (!matrix->entries[i])
        {
            fprintf(stderr, "matrix_create: failed to allocate row %d\n", i);
            for (int j = 0; j < i; j++)
            {
                free(matrix->entries[j]);
            }
            free(matrix->entries);
            free(matrix);
            return NULL;
        }
    }

    return matrix;
}

void matrix_free(Matrix *matrix)
{
    if (matrix)
    {
        if (matrix->entries)
        {
            for (int i = 0; i < matrix->rows; i++)
            {
                free(matrix->entries[i]);
            }
            free(matrix->entries);
        }
        free(matrix);
    }
}

void matrix_fill(Matrix *m, double n)
{
    if (!m)
        return;

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->entries[i][j] = n;
        }
    }
}

Matrix *matrix_zero(int rows, int cols)
{
    Matrix *m = matrix_create(rows, cols);
    if (!m)
    {
        fprintf(stderr, "matrix_zero: allocation failed\n");
        exit(EXIT_FAILURE);
    }
    matrix_fill(m, 0.0);
    return m;
}

Matrix *matrix_copy(Matrix *m)
{
    if (!m)
        return NULL;

    Matrix *mat = matrix_create(m->rows, m->cols);
    if (!mat)
    {
        fprintf(stderr, "matrix_copy: allocation failed\n");
        exit(EXIT_FAILURE);
    }

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
    if (!m)
    {
        printf("(null matrix)\n");
        return;
    }

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
    if (!m)
    {
        printf("Matrix: (null)\n");
        return;
    }
    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
}

void matrix_save(Matrix *m, FILE *file)
{
    if (!m || !file)
    {
        fprintf(stderr, "matrix_save: invalid arguments\n");
        return;
    }

    fprintf(file, "%d\n", m->rows);
    fprintf(file, "%d\n", m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            fprintf(file, "%.17g\n", m->entries[i][j]);
        }
    }
}

Matrix *matrix_load(FILE *file)
{
    if (!file)
    {
        fprintf(stderr, "matrix_load: file is NULL\n");
        return NULL;
    }

    char entry[MAXCHAR];

    if (!fgets(entry, MAXCHAR, file))
        return NULL;
    int rows = atoi(entry);

    if (!fgets(entry, MAXCHAR, file))
        return NULL;
    int cols = atoi(entry);

    Matrix *m = matrix_create(rows, cols);
    if (!m)
        return NULL;

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            if (!fgets(entry, MAXCHAR, file))
            {
                fprintf(stderr, "matrix_load: unexpected EOF while reading data\n");
                matrix_free(m);
                return NULL;
            }
            m->entries[i][j] = strtod(entry, NULL);
        }
    }
    return m;
}

int matrix_argmax(Matrix *m)
{
    // Expects a Mx1 matrix
    if (!m || m->rows <= 0 || m->cols != 1)
    {
        fprintf(stderr, "matrix_argmax: invalid matrix dimensions (%p, %d x %d)\n",
                (void *)m, m ? m->rows : -1, m ? m->cols : -1);
        exit(EXIT_FAILURE);
    }

    double max_score = m->entries[0][0];
    int max_idx = 0;

    for (int i = 1; i < m->rows; i++)
    {
        if (m->entries[i][0] > max_score)
        {
            max_score = m->entries[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}

Matrix *matrix_row(Matrix *m, int row_index)
{
    if (!m || row_index < 0 || row_index >= m->rows)
    {
        fprintf(stderr, "matrix_row: invalid row index %d\n", row_index);
        return NULL;
    }

    Matrix *row = matrix_create(1, m->cols);
    if (!row)
    {
        fprintf(stderr, "matrix_row: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int col = 0; col < m->cols; col++)
    {
        row->entries[0][col] = m->entries[row_index][col];
    }

    return row;
}

double matrix_mean_square_error(Matrix *output, Matrix *target)
{
    if (!output || !target)
    {
        fprintf(stderr, "matrix_mean_square_error: NULL matrices\n");
        exit(EXIT_FAILURE);
    }

    if (output->rows != target->rows || output->cols != target->cols)
    {
        fprintf(stderr, "Error: Matrices must have the same dimensions for MSE calculation (%d x %d vs %d x %d).\n",
                output->rows, output->cols, target->rows, target->cols);
        exit(EXIT_FAILURE);
    }

    double mse = 0.0;
    int total_elements = output->rows * output->cols;

    for (int i = 0; i < output->rows; i++)
    {
        for (int j = 0; j < output->cols; j++)
        {
            double error = output->entries[i][j] - target->entries[i][j];
            mse += error * error;
        }
    }

    mse /= (double)total_elements;
    return mse;
}

int matrix_check_dimensions(Matrix *m1, Matrix *m2)
{
    if (!m1 || !m2)
        return 0;
    return m1->rows == m2->rows && m1->cols == m2->cols;
}

static void matrix_seed_rng_once(void)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
}

void matrix_randomize(Matrix *m, double min, double max)
{
    if (!m)
        return;

    matrix_seed_rng_once();

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            double r = (double)rand() / (double)RAND_MAX;
            m->entries[i][j] = min + r * (max - min);
        }
    }
}

void matrix_xavier_randomize(Matrix *m, int input_size, int output_size)
{
    if (!m)
        return;

    matrix_seed_rng_once();

    double limit = sqrt(6.0 / (input_size + output_size));

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            double r = (double)rand() / (double)RAND_MAX;
            m->entries[i][j] = (2.0 * r - 1.0) * limit;
        }
    }
}

double matrix_sum_elements(Matrix *m)
{
    if (!m)
        return 0.0;

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
    if (!m1 || !m2 || !matrix_check_dimensions(m1, m2))
    {
        fprintf(stderr, "(matrix_add) Dimensions mismatch add: %dx%d %dx%d\n",
                m1 ? m1->rows : -1, m1 ? m1->cols : -1,
                m2 ? m2->rows : -1, m2 ? m2->cols : -1);
        exit(EXIT_FAILURE);
    }

    Matrix *m = matrix_create(m1->rows, m1->cols);
    if (!m)
    {
        fprintf(stderr, "matrix_add: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m2->cols; j++)
        {
            m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
        }
    }
    return m;
}

Matrix *matrix_subtract(Matrix *m1, Matrix *m2)
{
    if (!m1 || !m2 || !matrix_check_dimensions(m1, m2))
    {
        fprintf(stderr, "(matrix_subtract) Dimensions mismatch subtract: %dx%d %dx%d\n",
                m1 ? m1->rows : -1, m1 ? m1->cols : -1,
                m2 ? m2->rows : -1, m2 ? m2->cols : -1);
        exit(EXIT_FAILURE);
    }

    Matrix *m = matrix_create(m1->rows, m1->cols);
    if (!m)
    {
        fprintf(stderr, "matrix_subtract: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m2->cols; j++)
        {
            m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
        }
    }
    return m;
}

Matrix *matrix_multiply(Matrix *m1, Matrix *m2)
{
    if (!m1 || !m2 || m1->cols != m2->rows)
    {
        fprintf(stderr, "(matrix_multiply) Dimensions mismatch multiply: %dx%d %dx%d\n",
                m1 ? m1->rows : -1, m1 ? m1->cols : -1,
                m2 ? m2->rows : -1, m2 ? m2->cols : -1);
        exit(EXIT_FAILURE);
    }

    Matrix *result = matrix_create(m1->rows, m2->cols);
    if (!result)
    {
        fprintf(stderr, "matrix_multiply: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m2->cols; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < m1->cols; k++)
            {
                sum += m1->entries[i][k] * m2->entries[k][j];
            }
            result->entries[i][j] = sum;
        }
    }
    return result;
}

Matrix *matrix_dot(Matrix *m1, Matrix *m2)
{
    // Same semantics as matrix_multiply
    return matrix_multiply(m1, m2);
}

Matrix *matrix_apply(double (*func)(double), Matrix *m)
{
    if (!m || !func)
        return NULL;

    Matrix *mat = matrix_copy(m);
    if (!mat)
    {
        fprintf(stderr, "matrix_apply: allocation failed\n");
        exit(EXIT_FAILURE);
    }

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
    if (!m)
        return NULL;

    Matrix *mat = matrix_copy(m);
    if (!mat)
    {
        fprintf(stderr, "matrix_scale: allocation failed\n");
        exit(EXIT_FAILURE);
    }

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
    if (!m)
        return NULL;

    Matrix *mat = matrix_copy(m);
    if (!mat)
    {
        fprintf(stderr, "matrix_addScalar: allocation failed\n");
        exit(EXIT_FAILURE);
    }

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
    if (!m)
        return NULL;

    Matrix *mat = matrix_create(m->cols, m->rows);
    if (!mat)
    {
        fprintf(stderr, "matrix_transpose: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            mat->entries[j][i] = m->entries[i][j];
        }
    }
    return mat;
}

Matrix *matrix_softmax(Matrix *logits)
{
    if (!logits)
        return NULL;

    Matrix *probs = matrix_create(logits->rows, logits->cols);
    if (!probs)
    {
        fprintf(stderr, "matrix_softmax: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < logits->cols; j++)
    {
        double max_val = logits->entries[0][j];
        for (int i = 1; i < logits->rows; i++)
        {
            if (logits->entries[i][j] > max_val)
                max_val = logits->entries[i][j];
        }

        double sum_exp = 0.0;
        for (int i = 0; i < logits->rows; i++)
        {
            double e = exp(logits->entries[i][j] - max_val);
            probs->entries[i][j] = e;
            sum_exp += e;
        }

        if (sum_exp == 0.0)
            sum_exp = 1e-12;

        for (int i = 0; i < logits->rows; i++)
        {
            probs->entries[i][j] /= sum_exp;
        }
    }

    return probs;
}

double matrix_cross_entropy(Matrix *probs, Matrix *target)
{
    if (!probs || !target)
    {
        fprintf(stderr, "matrix_cross_entropy: NULL matrices\n");
        exit(EXIT_FAILURE);
    }

    if (probs->rows != target->rows || probs->cols != 1 || target->cols != 1)
    {
        fprintf(stderr, "matrix_cross_entropy: dimension mismatch %dx%d vs %dx%d\n",
                probs->rows, probs->cols, target->rows, target->cols);
        exit(EXIT_FAILURE);
    }

    double loss = 0.0;

    for (int i = 0; i < probs->rows; i++)
    {
        if (target->entries[i][0] == 1.0)
        {
            double p = probs->entries[i][0];
            if (p <= 0.0)
                p = 1e-12;
            loss = -log(p);
            break;
        }
    }

    return loss;
}
