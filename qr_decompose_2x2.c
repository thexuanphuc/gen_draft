#include <stdio.h>
#include <math.h>

// Function to print a 2x2 matrix
void print_matrix(const char* name, double M[2][2]) {
    printf("%s = [\n", name);
    for (int i = 0; i < 2; i++) {
        printf("  [ ");
        for (int j = 0; j < 2; j++) {
            printf("%f ", M[i][j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

// Compute the norm of a 2-element vector
double vector_norm(double v[2]) {
    return sqrt(v[0]*v[0] + v[1]*v[1]);
}

// Householder reflection for 2x2 matrix QR decomposition
void qr_decompose_2x2(double A[2][2], double Q[2][2], double R[2][2]) {
    // Copy A to R initially
    R[0][0] = A[0][0];
    R[1][0] = A[1][0];
    R[0][1] = A[0][1];
    R[1][1] = A[1][1];

    // Step 1: Compute Householder vector v to zero R[1][0]
    double x[2] = { R[0][0], R[1][0] };
    double norm_x = vector_norm(x);
    double sign = (x[0] >= 0) ? 1.0 : -1.0;
    double u1 = x[0] + sign * norm_x;
    double u2 = x[1];
    double norm_u = sqrt(u1*u1 + u2*u2);
    double v[2] = { u1/norm_u, u2/norm_u };

    // Step 2: Apply the Householder reflection to R from left: R = H * R 
    // where H = I - 2v v^T
    for (int j = 0; j < 2; j++) {
        double dot = v[0]*R[0][j] + v[1]*R[1][j];
        R[0][j] -= 2 * v[0] * dot;
        R[1][j] -= 2 * v[1] * dot;
    }

    // Step 3: Form Q = H, Q is orthogonal
    // Q = I - 2*v*v^T
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (i == j) {
                Q[i][j] = 1 - 2 * v[i]*v[j];
            } else {
                Q[i][j] = -2 * v[i]*v[j];
            }
        }
    }
}

int main() {
    double A[2][2] = {
        {4, 2},
        {3, 1}
    };

    double Q[2][2], R[2][2];

    qr_decompose_2x2(A, Q, R);

    print_matrix("Q", Q);
    print_matrix("R", R);

    return 0;
}
