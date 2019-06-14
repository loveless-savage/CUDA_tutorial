/*	now let's drop some CUDA in there!
 *	instead of running all on the CPU, we will hand off the task of adding floats to the GPU
 *	for now, we'll stick with one thread
 *	compile with CUDA compiler:
		nvcc cuda1.cu -o cuda1
*/
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//

#include <cstdio>
#include <cmath>

int arraySize = 1<<20; // basically a million

// function to add them together
__global__																/*/*/	// this makes addArrays() accessible to the GPU
void addArrays (int arraySize, float *add1, float *add2, float *sum){			// addArrays() is now considered a kernel
	for (int i=0; i<arraySize; i++){
		sum[i] = add1[i] + add2[i];
	}
}

// all the action
int main(){

	// three arrays; we will add the first two to sum[]
	printf("initializing arrays\n");
	float *add1, *add2, *sum;											/*/*/	// CUDA allows us to set up a memory space
	cudaMallocManaged( &add1, arraySize*sizeof(float) );				/*/*/	// accessible by the CPU and GPU alike
	cudaMallocManaged( &add2, arraySize*sizeof(float) );				/*/*/	// cudaMallocManaged(), like malloc(),
	cudaMallocManaged( &sum,  arraySize*sizeof(float) );				/*/*/	// returns pointers usable by both devices

	// fill first two arrays before the CUDA starts
	for (int i=0; i<arraySize; i++){
		add1[i] = 1.0;
		add2[i] = 2.0;
	}
	printf("arrays done. prepare for adding\n");

	// parallelization happens here
	addArrays<<<1,1>>>(arraySize, add1,add2,sum);						/*/*/	// <<<1,1>>> tells CPU to give task to the GPU
																		/*/*/	// the 1's will be explained later, but in
	// wait for all threads to complete on the GPU								// this case it means just one thread
	cudaDeviceSynchronize();											/*/*/	// then we wait for all GPU threads to finish calculating	
	printf("adding complete.\t");												// now the CPU is back in charge

	// check for accuracy- what's the biggest mistake?
	float maxError = 0.0;
	for (int i=0; i<arraySize; i++){
		// check each array index for value and store the greater deviation from 3.0
		maxError = fmax(maxError, fabs(sum[i]-3.0));
	}
	printf("max error = %f\n",maxError);

	// free memory
	cudaFree(add1);														/*/*/	// we need to use cudaFree()
	cudaFree(add2);														/*/*/	// instead of delete []
	cudaFree(sum);														/*/*/	// because it's shared CUDA memory

	return 0;
}
