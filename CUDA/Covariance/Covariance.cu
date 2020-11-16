#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

/* Problem size */
#define M 1024
#define N 1024

#define FLOAT_N 3214212.01

void init_arrays(double* data)
{
	int i, j;

	//Αρχικοποίηση του πίνακα data
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			data[i*N + j] = ((double) (i+1)*(j+1)) / M;
		}
	}
}

__global__ void dataElementAverage(double* data, double* average)
{
	int	i, j;

	//Κάθε thread θα γράψει σε συγκεκριμένο κελί μνήμης
	// block/grid   using block   using thread
  i = blockDim.x * blockIdx.x + threadIdx.x;

  //Δημιουργία του πίνακα average
  if (i < M) {
    average[i] = 0.0;
		//Πρόσθεση τιμών στήλης του data
  	for (j = 0; j < N; j++) {
      average[i] += data[i + j*M];
  	}
		//Εύρεση ΜΟ της στήλης
  	average[i] /= FLOAT_N;
  }
}

__global__ void data_minus_average(double* data, double* average)
{
	int	i, j;

  i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < M) {
		//Αφαίρεση του ΜΟ από το κάθε στοιχείο αυτής της στήλης
    for (j = 0; j < M; j++) {
			data[i + j*M] -= average[j];
		}
  }
}

__global__ void covariance(double* data, double* symmat)
{
	int	i,j,j2;

	i = blockDim.x * blockIdx.x + threadIdx.x;

	//Πολλαπλασιασμός του πίνακα data με τον ανάστροφο του
	//Αν o αριθμός των threads δεν ξεπερνά τον αριθμό των στοιχείων του πίνακα
	if (i < M) {
			//Υπολογισμός γραμμής του πίνακα symmat
			for (j = 0; j < M; j++) {
							//Αρχικοποίηση του πίνακα symmat
		       		symmat[i*M + j] = 0.0;
				for (j2 = 0; j2 < N; j2++) {
					//Υπολογισμός στοιχείου του πίνακα symmat
																//γραμμή				//στήλη
					symmat[i*M + j] += data[j2 + i*M] * data[j2 + j*M];
				}
				//Αντιγραφή στοιχείων στις "απέναντι" θέσης του πίνακα για συμμετρία
	      symmat[i + j*M] = symmat[i*M + j];
	    }
	}
}

int main(int argc, char *argv[])
{

	//Δημιουργία χρονομετρητή για τον kernel
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	double		*data, *symmat, *average, *d_data, *d_symmat, *d_average;


	//Δέσμευση μνήμης σε host και device
	data = (double*)malloc(M*N*sizeof(double));
	symmat = (double*)malloc(M*M*sizeof(double));
	average = (double*)malloc(M*sizeof(double));
  cudaMalloc((void**)&d_data, M*N*sizeof(double));
  cudaMalloc((void**)&d_symmat, M*M*sizeof(double));
  cudaMalloc((void**)&d_average, M*sizeof(double));

	//initialize the array
	init_arrays(data);

  /*printf("\nInitialized array data:\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			printf("%lf ", data[i*M + j]);
		}
		printf("\n");
	}
	printf("\n");*/

	//Αντιγραφή απαραίτητων πινάκων στην GPU
  cudaMemcpy(d_data, data, M*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_symmat, symmat, M*M*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_average, average, M*sizeof(double), cudaMemcpyHostToDevice);

	//Δημιουργία 1D πλέγματος και block
  dim3 Grid((M-1)/2+1,1,1);
  dim3 Block(2,1,1);

	//Αρχή χρονομέτρησης
	cudaEventRecord(start);
	//Κλήση kernel
	dataElementAverage<<<Grid,Block>>>(d_data, d_average);
	//Τέλος χρονομέτρησης
	cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds,start,stop);
  printf("Kernel one elapsed time: %f seconds.\n",milliseconds/1000);

	//Αντιγραφή του πίνακα average από την κάρτα γραφικών στον host
	cudaMemcpy(average, d_average, M*sizeof(double), cudaMemcpyDeviceToHost);

	/*printf("\nAverage array:\n");
	for (int i = 0; i < N; i++) {
		printf("%lf ", average[i]);
	}
	printf("\n");*/

  cudaEventRecord(start);
	data_minus_average<<<Grid,Block>>>(d_data, d_average);
	cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds,start,stop);
  printf("Kernel two elapsed time: %f seconds.\n",milliseconds/1000);

	cudaMemcpy(data, d_data, M*N*sizeof(double), cudaMemcpyDeviceToHost);

	/*printf("\nData sub average array:\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			printf("%lf ", data[i + j*M]);
		}
		printf("\n");
	}
	printf("\n");*/

  cudaEventRecord(start);
	covariance<<<Grid,Block>>>(d_data, d_symmat);
	cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds,start,stop);
  printf("Kernel three elapsed time: %f seconds.\n",milliseconds/1000);

	cudaMemcpy(symmat, d_symmat, M*N*sizeof(double), cudaMemcpyDeviceToHost);

	/*printf("\nsymmat array:\n");
	for (int j1 = 0; j1 < M; j1++) {
		for (int j2 = 0; j2 < N; j2++) {
    	printf("%lf ", symmat[j1*M + j2]);
    }
		printf("\n");
	}
	printf("\n");*/

  cudaFree(d_data);
  cudaFree(d_symmat);
  cudaFree(d_average);
	free(data);
	free(symmat);
	free(average);

  return 0;
}
