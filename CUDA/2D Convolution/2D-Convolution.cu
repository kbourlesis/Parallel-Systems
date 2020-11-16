#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define NI 4096
#define NJ 4096

__global__ void Convolution(double* A, double* B)
{
	int i, j;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	//Κάθε thread θα γράψει σε συγκεκριμένο κελί μνήμης
	// block/grid   using block   using thread
  i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  j = blockDim.y * blockIdx.y + threadIdx.y + 1;

	//Convolution
	if(i < NI && j < NJ) {
    B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
          + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)]
          + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
	}
}

void init(double* A)
{
	int i, j;
//Αρχικοποίηση του πίνακα Α
	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i*NJ + j] = (double)rand()/RAND_MAX;
        	}
    	}
}

int main(int argc, char *argv[])
{

	//Δημιουργία χρονομετρητή για τον kernel
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	double		*A, *B, *d_A, *d_B;

	//Δέσμευση μνήμης σε host και device
	A = (double*)malloc(NI*NJ*sizeof(double));
	B = (double*)malloc(NI*NJ*sizeof(double));
  cudaMalloc((void**)&d_A,NI*NJ*sizeof(double));
  cudaMalloc((void**)&d_B,NI*NJ*sizeof(double));

	//initialize the array
	init(A);

  /*printf("\nArray A:\n");
  for (int i = 0; i < NI; ++i) {
		for (int j = 0; j < NJ; ++j) {
			printf("%lf ",A[i*NJ + j]);
		}
		printf("\n");
	}
	printf("\n");*/

	//Αντιγραφή του πίνακα Α στην κάρτα γραφικών (μεταβλητή d_A)
	cudaMemcpy(d_A, A, NI*NJ*sizeof(double), cudaMemcpyHostToDevice);

	//Δημιουργία 2D πλέγματος και block
  dim3 Grid((NI-1)/2+1,(NJ-1)/2+1,1);
  dim3 Block(2,2,1);

	//Αρχή χρονομέτρησης
  cudaEventRecord(start);
	//Κλήση kernel
	Convolution<<<Grid,Block>>>(d_A, d_B);
	//Τέλος χρονομέτρησης
	cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds,start,stop);
  printf("Elapsed Time: %f seconds.\n",milliseconds/1000);

	//Αντιγραφή του πίνακα B από την κάρτα γραφικών στον host
	cudaMemcpy(B, d_B, NI*NJ*sizeof(double),cudaMemcpyDeviceToHost);

  /*printf("\nArray B:\n");
  for (int i = 0; i < NI; ++i) {
		for (int j = 0; j < NJ; ++j) {
			printf("%lf ",B[i*NJ + j]);
		}
		printf("\n");
	}
	printf("\n");*/

	//Απελευθέρωση μνήμης σε host και device
  cudaFree(d_A);
  cudaFree(d_B);
	free(A);
	free(B);

	return 0;
}
