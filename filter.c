#include <stddef.h>

// Kalman filter update for 2D state, 1D measurement
// All matrices are 1D float arrays, no structs used

void kalman_filter_update(
    float *A,      // [A21]
    float *P,      // [P11, P12, P21, P22]
    float *Q,      // [Q11, Q22]
    float dt,
    float x_pred,
    float vx_pred,
    float B2u_in,
    float x_step,
    float y_meaUD,
    float R,
    float *x_out,  // [x0, x1] output
    float *P_out   // [P11, P12, P21, P22] output
) {
    // Unpack matrices
    float A21 = A[0];
    float P11 = P[0], P12 = P[1], P21 = P[2], P22 = P[3];
    float Q11 = Q[0], Q22 = Q[1];

    // Prediction step
    float x_temp1 = dt * vx_pred + x_pred + x_step;
    float x_temp2 = A21 * x_pred + B2u_in + vx_pred;

    float P_pred11 = P11 + P21 * dt + Q11 + dt * (P12 + P22 * dt);
    float P_pred12 = A21 * (P11 + P21 * dt) + P12 + P22 * dt;
    float P_pred21 = A21 * P11 + P21 + dt * (A21 * P12 + P22);
    float P_pred22 = A21 * (P12 + A21 * P11 + P21) + P22 + Q22;

    // Innovation
    float S = P11 + P21 * dt + Q11 + R + dt * (P12 + P22 * dt);

    // Kalman gain
    float K0 = P_pred11 / S;
    float K1 = P_pred21 / S;


    // Update state using the provided formula
    x_out[0] = -K1 * ( x_temp1 - y_meaUD) + x_temp1;
    x_out[1] = -K0 * ( x_temp1 - y_meaUD) + x_temp2 + vx_step;

    // Update covariance

    P_out[0] = K1 * K1 * R + P_pred11 * (K1 - 1) * (K1 - 1);
    P_out[1] = K1 * K2 * R + K2 * P_pred11 * (K1 - 1) - P_pred12 * (K1 - 1);
    P_out[2] = K1 * K2 * R + (K1 - 1) * (K2 * P_pred11 - P_pred21);
    P_out[3] = K2 * K2 * R - K2 * P_pred12 + K2 * (K2 * P_pred11 - P_pred21) + P_pred22;
}