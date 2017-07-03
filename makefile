gaspi.out: gaspi.c
	gcc -o gaspi.out gaspi.c  -std=gnu99 /opt/GPI2/lib64/libGPI2.a -lpthread -I/home/oliver/GaspiProjekt/HelloWorld -O3 -Wall -lm -fno-tree-vectorize
openmp.out: openmp.c
	gcc -o openmp.out openmp.c  -std=gnu99  -O1 -Wall -lm -fno-tree-vectorize
mpi.out: mpi.c
	mpicc -o mpi.out mpi.c -std=c99 -O3 -Wall -lm -fno-tree-vectorize
