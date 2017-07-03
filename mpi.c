#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#define G (4*M_PI*M_PI)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define RIGHT(me, n_proc) ((me+1) % n_proc)
#define LEFT(me, n_proc) ((me - 1 + n_proc) % n_proc)
typedef struct {
    double x, y, z, m;
} Particle_send_data;

typedef struct {
    double vx, vy, vz, ax, ay, az;
} Particle_local_data;

MPI_Datatype mpi_particle_send_data_type;

void do_simulation(const double dt, const int n_steps, const int n_bodies, const int n_proc, const int me_proc);
void move_part(Particle_send_data* const own_particles_send_data, Particle_local_data* const own_particles_local_data, const int n_bodies, const double dt);

void calculate_acc_total(Particle_send_data* const own_particles_send_data, Particle_local_data* const own_particles_local_data, const int n_bodies, const int n_proc, const int me_proc);

void calculate_acc_local(Particle_send_data* const own_particles_send_data, Particle_local_data* const own_particles_local_data, const int n_bodies);

void calculate_acc_extern(Particle_send_data* const restrict own_particles_send_data, Particle_local_data* const own_particles_local_data, const Particle_send_data* const restrict ext_particles_send_data, int n_bodies);

void set_initial_conditions(Particle_send_data* own_particle_send_data, Particle_local_data* own_particle_local_data, const int n_bodies, const int me_proc);

void sum_values(const Particle_send_data* const own_particle_send_data, double* sum,int size) {
    sum[0] = 0;
    sum[1] = 0;
    sum[2] = 0;
    for(int i = 0;i < size; i++) {
        sum[0] += own_particle_send_data[i].x;
        sum[1] += own_particle_send_data[i].y;
        sum[2] += own_particle_send_data[i].z;
    }
    return;
}

int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    int me_proc, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &me_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    printf("proc: %i/ %i started\n", me_proc,n_proc);
    const int n_steps = 20000;
    const int n_bodies = 100;
    const double dt = 0.05;
    const int nitems=4;
    int blocklengths[10] = {1,1,1,1};
    MPI_Datatype types[10] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[4];

    offsets[0] = offsetof(Particle_send_data, x);
    offsets[1] = offsetof(Particle_send_data, y);
    offsets[2] = offsetof(Particle_send_data, z);
    offsets[3] = offsetof(Particle_send_data, m);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_particle_send_data_type);
    MPI_Type_commit(&mpi_particle_send_data_type);


    do_simulation(dt, n_steps, n_bodies, n_proc, me_proc);

    MPI_Type_free(&mpi_particle_send_data_type);
    MPI_Finalize();
    return 0;
}

void do_simulation(const double dt, const int n_steps, const int n_bodies, const int n_proc, const int me_proc) {
    Particle_send_data* own_particle_send_data = (Particle_send_data*) malloc(3*sizeof(Particle_send_data)*n_bodies);
    Particle_local_data* own_particle_local_data = (Particle_local_data*) malloc(sizeof(Particle_local_data)*n_bodies);

    set_initial_conditions(own_particle_send_data, own_particle_local_data, n_bodies, me_proc);

    time_t start = clock();
    for(int i = 0; i < n_steps; i++) {
        calculate_acc_total(own_particle_send_data, own_particle_local_data, n_bodies, n_proc, me_proc);
        move_part(own_particle_send_data, own_particle_local_data,n_bodies, dt);
    }


    time_t end = clock();
    double loc_sum[3];
    sum_values(own_particle_send_data, loc_sum, n_bodies);
    double glob_sum[3] = {0,0,0};
    MPI_Allreduce(loc_sum, glob_sum, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(me_proc == 0 || 1) {
        printf("x: %f, y: %f, z: %f, xloc: %f, yloc: %f, zloc: %f, time: %f\n",glob_sum[0], glob_sum[1], glob_sum[2], loc_sum[0], loc_sum[1], loc_sum[2],(end-start)/(double)CLOCKS_PER_SEC );
    }
    return;
}



void calculate_acc_total(Particle_send_data* const own_particles_send_data, Particle_local_data* const own_particles_local_data, const int n_bodies, const int n_proc, const int me_proc) {
    Particle_send_data* ext_particles_send_data_1 = &own_particles_send_data[n_bodies];
    Particle_send_data* ext_particles_send_data_2 = &own_particles_send_data[2*n_bodies];
    if(n_proc==1) {
        calculate_acc_local(own_particles_send_data, own_particles_local_data, n_bodies);
    } else if(n_proc ==2) {
        MPI_Request send_own_part;
        MPI_Request recv_1_part;
        MPI_Irecv(ext_particles_send_data_1, n_bodies, mpi_particle_send_data_type, LEFT(me_proc, n_proc), 0, MPI_COMM_WORLD, &recv_1_part);
        MPI_Isend(own_particles_send_data, n_bodies, mpi_particle_send_data_type, RIGHT(me_proc, n_proc), 0, MPI_COMM_WORLD, &send_own_part);
        calculate_acc_local(own_particles_send_data, own_particles_local_data, n_bodies);
        MPI_Wait(&recv_1_part,  MPI_STATUSES_IGNORE);
        calculate_acc_extern(own_particles_send_data, own_particles_local_data,ext_particles_send_data_1, n_bodies);
        MPI_Wait(&send_own_part,  MPI_STATUSES_IGNORE);
    }  else {
        MPI_Request send_part[2];
        MPI_Request recv_part[2];

        MPI_Irecv(ext_particles_send_data_1, n_bodies, mpi_particle_send_data_type, LEFT(me_proc, n_proc), 0, MPI_COMM_WORLD, &recv_part[0]);
        MPI_Irecv(ext_particles_send_data_2, n_bodies, mpi_particle_send_data_type, LEFT(me_proc, n_proc), 1, MPI_COMM_WORLD, &recv_part[1]);

        MPI_Isend(own_particles_send_data, n_bodies, mpi_particle_send_data_type, RIGHT(me_proc, n_proc), 0, MPI_COMM_WORLD, &send_part[0]);
        calculate_acc_local(own_particles_send_data, own_particles_local_data, n_bodies);
        MPI_Wait(&recv_part[0],  MPI_STATUSES_IGNORE);
        MPI_Wait(&send_part[0],  MPI_STATUSES_IGNORE);
        MPI_Isend(ext_particles_send_data_1, n_bodies, mpi_particle_send_data_type, RIGHT(me_proc, n_proc), 1, MPI_COMM_WORLD, &send_part[1]);

        calculate_acc_extern(own_particles_send_data, own_particles_local_data,ext_particles_send_data_1, n_bodies);

        Particle_send_data* ext_particles_send_data_send = ext_particles_send_data_1;
        Particle_send_data* ext_particles_send_data_recv = ext_particles_send_data_2;
        for(int i=2;i<n_proc-1;i++) {
            Particle_send_data* tmp = ext_particles_send_data_send;
            ext_particles_send_data_send = ext_particles_send_data_recv;
            ext_particles_send_data_recv = tmp;
            MPI_Wait(&recv_part[(i-1)%2],  MPI_STATUSES_IGNORE);
            MPI_Wait(&send_part[(i-1)%2],  MPI_STATUSES_IGNORE);

            MPI_Isend(ext_particles_send_data_send, n_bodies, mpi_particle_send_data_type, RIGHT(me_proc, n_proc), i, MPI_COMM_WORLD, &send_part[i%2]);
            MPI_Irecv(ext_particles_send_data_recv, n_bodies, mpi_particle_send_data_type, LEFT(me_proc, n_proc), i, MPI_COMM_WORLD, &recv_part[i%2]);

            calculate_acc_extern(own_particles_send_data, own_particles_local_data,ext_particles_send_data_send, n_bodies);

        }
        MPI_Wait(&recv_part[(n_proc-2)%2],  MPI_STATUSES_IGNORE);

        calculate_acc_extern(own_particles_send_data, own_particles_local_data,ext_particles_send_data_recv, n_bodies);
        MPI_Wait(&send_part[(n_proc-2)%2],  MPI_STATUSES_IGNORE);
    }
    return;
}

void calculate_acc_local(Particle_send_data* const own_particles_send_data, Particle_local_data* const own_particles_local_data, const int n_bodies) {
    for(int i = 0;i <n_bodies;i++) {
        for(int j = i+1; j<n_bodies; j++) {
            double dx = own_particles_send_data[i].x - own_particles_send_data[j].x;
            double dy = own_particles_send_data[i].y - own_particles_send_data[j].y;
            double dz = own_particles_send_data[i].z - own_particles_send_data[j].z;

            double rabs = sqrt(dx*dx + dy*dy + dz*dz);
            double rabs_3 = 1 / (rabs*rabs*rabs);
            double i_multiply = G * rabs_3 * own_particles_send_data[j].m;
            double j_multiply = G * rabs_3 * own_particles_send_data[i].m;
            //unsure !!!!
            own_particles_local_data[i].ax -=  i_multiply * dx;
            own_particles_local_data[i].ay -= i_multiply * dy;
            own_particles_local_data[i].az -= i_multiply * dz;

            own_particles_local_data[j].ax += j_multiply * dx;
            own_particles_local_data[j].ay += j_multiply * dy;
            own_particles_local_data[j].az += j_multiply * dz;
        }
    }
    return;
}

void calculate_acc_extern(Particle_send_data* const restrict own_particles_send_data, Particle_local_data* const own_particles_local_data, const Particle_send_data* const restrict ext_particles_send_data, int n_bodies) {
    for(int i = 0; i < n_bodies; i++) {
        for(int j = 0; j < n_bodies; j++) {
            double dx = own_particles_send_data[i].x - ext_particles_send_data[j].x;
            double dy = own_particles_send_data[i].y - ext_particles_send_data[j].y;
            double dz = own_particles_send_data[i].z - ext_particles_send_data[j].z;

            double rabs = sqrt(dx*dx + dy*dy + dz*dz);
            double rabs_3 = 1 / (rabs*rabs*rabs);
            double i_multiply = G * rabs_3 * ext_particles_send_data[j].m;
            //unsure !!!!
            own_particles_local_data[i].ax -=  i_multiply * dx;
            own_particles_local_data[i].ay -= i_multiply * dy;
            own_particles_local_data[i].az -= i_multiply * dz;
        }
    }
    return;
}



void move_part(Particle_send_data* const own_particles_send_data, Particle_local_data* const own_particles_local_data, const int n_bodies, const double dt) {
    for(int i = 0; i< n_bodies; i++) {
        own_particles_local_data[i].vx += own_particles_local_data[i].ax * dt;
        own_particles_local_data[i].vy += own_particles_local_data[i].ay * dt;
        own_particles_local_data[i].vz += own_particles_local_data[i].az * dt;
        own_particles_send_data[i].x += own_particles_local_data[i].vx * dt;
        own_particles_send_data[i].y += own_particles_local_data[i].vy * dt;
        own_particles_send_data[i].z += own_particles_local_data[i].vz * dt;
        own_particles_local_data[i].ax = 0;
        own_particles_local_data[i].ay = 0;
        own_particles_local_data[i].az = 0;
    }
}

void set_initial_conditions(Particle_send_data* own_particle_send_data, Particle_local_data* own_particle_local_data, const int n_bodies, const int me_proc) {
    if(n_bodies ==1) {
        if(me_proc==0) {
            own_particle_send_data[0].x = 1;
            own_particle_send_data[0].y = 0;
            own_particle_send_data[0].z = 0;
            own_particle_local_data[0].vx = 0;
            own_particle_local_data[0].vy = 5+2*M_PI;
            own_particle_local_data[0].vz = 1;
            own_particle_send_data[0].m = 0.000001;
        } else{
            own_particle_send_data[0].x = 0;
            own_particle_send_data[0].y = 0;
            own_particle_send_data[0].z = 0;
            own_particle_local_data[0].vx = 0;
            own_particle_local_data[0].vy = 5;
            own_particle_local_data[0].vz = 1;
            own_particle_send_data[0].m = 1;
        }
    } else {
        for(int i = 0; i< n_bodies;i++ ){
            int j_global = n_bodies * me_proc +i;
            own_particle_send_data[i].x = 500*j_global*pow(-1,j_global)*sin(j_global)+5*j_global;
            own_particle_send_data[i].y = -200*j_global*pow(-1,j_global)*cos(j_global)+5*j_global;
            own_particle_send_data[i].z = 0;
            own_particle_local_data[i].vx = 10*j_global*j_global*pow(-1,j_global);
            own_particle_local_data[i].vy = -5*j_global*j_global*pow(-1,j_global);
            own_particle_local_data[i].vz = pow(-1,j_global);
            own_particle_local_data[i].ax = 0;
            own_particle_local_data[i].ay = 0;
            own_particle_local_data[i].az = 0;
            own_particle_send_data[i].m = 1000;
        }
    }
    return;
}
