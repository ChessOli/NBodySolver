//#define withOpenMP
#ifdef withOpenMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define G (4*M_PI*M_PI)


void do_simulation(const double dt, const int n_steps, const int n_bodies);
void move_part(double* restrict x, double *restrict y, double *restrict z,
               double* restrict vx, double *restrict vy, double *restrict vz,
               double* restrict ax, double* restrict ay, double *restrict az,
               const int n_bodies, const double dt);

void calculate_acc(const double* restrict x, const double *restrict y, const double *restrict z,
                   double* restrict ax, double* restrict ay, double* restrict az,
                   const double* restrict m,
                   const int n_bodies);

void set_initial_conditions(double* restrict x, double *restrict y, double *restrict z,
                            double* restrict vx, double *restrict vy, double *restrict vz,
                            double* restrict m, const int n_bodies);

double sum_values(const double* restrict x, int size) {
    double sum = 0;
    for(int i = 0;i < size; i++) {
        sum += x[i];
    }
    return sum;
}

int main(int argc, char**argv)
{
    const int n_steps = 200;
    const int n_bodies = 300;
    const double dt = 0.05;
#ifdef withOpenMP
    if(argc==1) {
        omp_set_num_threads(2);
     } else {
        int num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    }
#endif
    do_simulation(dt, n_steps, n_bodies);
    return 0;
}

void do_simulation(const double dt, const int n_steps, const int n_bodies) {
    double *x = malloc(n_bodies*sizeof(double));
    double *y = malloc(n_bodies*sizeof(double));
    double *z = malloc(n_bodies*sizeof(double));

    double *vx = malloc(n_bodies*sizeof(double));
    double *vy = malloc(n_bodies*sizeof(double));
    double *vz = malloc(n_bodies*sizeof(double));

#ifdef withOpenMP
    int n_threads = omp_get_max_threads();
    double *ax = calloc(n_threads * n_bodies, sizeof(double) );
    double *ay = calloc(n_threads * n_bodies, sizeof(double));
    double *az = calloc(n_threads * n_bodies, sizeof(double));
#else
    double *ax = calloc(n_bodies, sizeof(double) );
    double *ay = calloc(n_bodies, sizeof(double));
    double *az = calloc(n_bodies, sizeof(double));
#endif
    double *m = malloc(n_bodies*sizeof(double));

    set_initial_conditions(x, y , z, vx, vy, vz, m, n_bodies);
    time_t start = clock();
    for(int i = 0; i < n_steps; i++) {
        calculate_acc(x, y, z, ax, ay, az, m, n_bodies);

        move_part(x, y, z, vx, vy, vz, ax, ay, az, n_bodies, dt);
    }

    time_t end = clock();
    double x_sum = sum_values(x, n_bodies);
    double y_sum = sum_values(y, n_bodies);
    double z_sum = sum_values(z, n_bodies);
    printf("x: %f, y: %f, z: %f , time: %f\n",x_sum,y_sum,z_sum, (end-start)/(double)CLOCKS_PER_SEC );

    free(x);
    free(y);
    free(z);

    free(vx);
    free(vy);
    free(vz);

    free(ax);
    free(ay);
    free(az);
}


void calculate_acc(const double* restrict x, const double *restrict y, const double *restrict z,
                   double* restrict ax, double* restrict ay, double *restrict az,
                   const double* restrict m,
                   const int n_bodies) {
#ifdef withOpenMP
    #pragma omp parallel for schedule(static, 4)
    for(int i = 0; i < n_bodies; i++) {
        int thread_id = omp_get_thread_num();
        for(int j = i+1; j<n_bodies; j++) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];

            double rabs = sqrt(dx*dx + dy*dy + dz*dz);
            double rabs_3 = 1 / (rabs*rabs*rabs);
            double i_multiply = G * rabs_3 * m[j];
            double j_multiply = G * rabs_3 * m[i];
            //unsure !!!!
            ax[i +n_bodies*thread_id] -=  i_multiply * dx;
            ay[i +n_bodies*thread_id] -= i_multiply * dy;
            az[i +n_bodies*thread_id] -= i_multiply * dz;

            ax[j +n_bodies*thread_id] += j_multiply * dx;
            ay[j +n_bodies*thread_id] += j_multiply * dy;
            az[j +n_bodies*thread_id] += j_multiply * dz;
        }
    }

    int n_threads = omp_get_max_threads();
    #pragma omp parallel for schedule (static,4)
    for(int i = 0; i<n_bodies;i++) {
        for(int j = 1; j< n_threads;j++) {
            ax[i] +=ax[i+j*n_bodies];
            ay[i] +=ay[i+j*n_bodies];
            az[i] +=az[i+j*n_bodies];
        }
    }
#else
    for(int i = 0;i <n_bodies;i++) {
        for(int j = i+1; j<n_bodies; j++) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];

            double rabs = sqrt(dx*dx + dy*dy + dz*dz);
            double rabs_3 = 1 / (rabs*rabs*rabs);
            double i_multiply = G * rabs_3 * m[j];
            double j_multiply = G * rabs_3 * m[i];
            //unsure !!!!
            ax[i] -=  i_multiply * dx;
            ay[i] -= i_multiply * dy;
            az[i] -= i_multiply * dz;

            ax[j] += j_multiply * dx;
            ay[j] += j_multiply * dy;
            az[j] += j_multiply * dz;
        }
    }
#endif
    return;
}

void move_part(double* restrict x, double *restrict y, double *restrict z,
               double* restrict vx, double* restrict vy, double *restrict vz,
               double* restrict ax, double* restrict ay, double *restrict az,
               const int n_bodies, const double dt) {
    #pragma omp parallel for
    for(int i = 0; i< n_bodies; i++) {
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        vz[i] += az[i] * dt;
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
#ifndef withOpenMP
        ax[i] = 0;
        ay[i] = 0;
        ay[i] = 0;
#endif
    }
#ifdef withOpenMP
    int n_threads = omp_get_max_threads();
    #pragma omp parallel for
    for(int i=0;i<n_bodies * n_threads;i++) {
        ax[i] = 0;
        ay[i] = 0;
        ay[i] = 0;
    }

#endif
}

void set_initial_conditions(double* restrict x, double *restrict y, double *restrict z,
                            double* restrict vx, double *restrict vy, double *restrict vz,
                            double* restrict m, const int n_bodies) {
    if(n_bodies ==2) {
        x[0] = 1;
        y[0] = 0;
        z[0] = 0;
        vx[0] = 0;
        vy[0] = 5+2*M_PI;
        vz[0] = 1;
        m[0] = 0.000001;
        x[1] = 0;
        y[1] = 0;
        z[1] = 0;
        vx[1] = 0;
        vy[1] = 5;
        vz[1] = 1;
        m[1] = 1;
    } else {
    #pragma omp parallel
        for(int i = 0; i< n_bodies;i++ ){
            x[i] = 500*i*pow(-1,i)*sin(i)+5*i;
            y[i] = -200*i*pow(-1,i)*cos(i)+5*i;
            z[i] = 0;
            vx[i] = 10*i*i*pow(-1,i);
            vy[i] = -5*i*i*pow(-1,i);
            vz[i] = pow(-1,i);
            m[i] = 1000;
        }
    }
}
