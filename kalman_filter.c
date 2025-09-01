void kalman_update(
    double x0, double x1, 
    double P00, double P01, double P10, double P11, 
    double H00, double H01, double H10, double H11, 
    double R00, double R01, double R10, double R11, 
    double z0, double z1, 
    double* x_new0, double* x_new1, 
    double* P_new00, double* P_new01, double* P_new10, double* P_new11) {

    double t0 = -H10*x0 - H11*x1 + z1;
    double t1 = H00*P00;
    double t2 = H01*P10 + t1;
    double t3 = H01*P11;
    double t4 = H00*P01 + t3;
    double t5 = H00*t2 + H01*t4 + R00;
    double t6 = H10*P00;
    double t7 = H11*P10 + t6;
    double t8 = H11*P11;
    double t9 = H10*P01 + t8;
    double t10 = H10*t7 + H11*t9 + R11;
    double t11 = H00*t7;
    double t12 = H01*t9;
    double t13 = H10*t2;
    double t14 = H11*t4;
    double t15 = 1.0/(t10*t5 - (R01 + t13 + t14)*(R10 + t11 + t12));
    double t16 = t15*(H11*P01 + t6);
    double t17 = -R01 - t13 - t14;
    double t18 = t15*(H01*P01 + t1);
    double t19 = t16*t5 + t17*t18;
    double t20 = -H00*x0 - H01*x1 + z0;
    double t21 = -R10 - t11 - t12;
    double t22 = t10*t18 + t16*t21;
    double t23 = t15*(H10*P10 + t8);
    double t24 = t15*(H00*P10 + t3);
    double t25 = t17*t24 + t23*t5;
    double t26 = t10*t24 + t21*t23;
    double t27 = -H01*t22 - H11*t19;
    double t28 = -H00*t22 - H10*t19 + 1;
    double t29 = -H00*t26 - H10*t25;
    double t30 = -H01*t26 - H11*t25 + 1;

    *x_new0 = t0*t19 + t20*t22 + x0;
    *x_new1 = t0*t25 + t20*t26 + x1;
    *P_new00 = P00*t28 + P10*t27;
    *P_new01 = P01*t28 + P11*t27;
    *P_new10 = P00*t29 + P10*t30;
    *P_new11 = P01*t29 + P11*t30;
}
