/*	so we know how to interface with blocks/threads conceptually.
 *	now, the question is: what is the syntax?
 *	compile with cuda compiler:
		nvcc cuda4.cu -o cuda4
*/
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//

#include <cstdio>
#include <cmath>

int arraySize = 1<<20; // basically a million

// function to add them together
__global__
void addArrays (int arraySize, float *add1, float *add2, float *sum){			// now each copy of our kernel doesn't have to handle all 1M operations!
/*	so how do you keep track of all these kernels?
 *	each thread has an ID number for its location within its own block
 *	this variable is called @@@ threadIdx.x @@@
 *	each block also has an ID number for its location within the grid
 *	this variable is called @@@ blockIdx.x @@@
 *	dimensions of blocks are @@@ blockDim.x @@@
 *	like this:
	drawing		{	|----------------|----------------|----------------|----------------|
	labels		{	            thread[^]        block[________________]
	gridDim.x	{	                                                                   = 4
	blockIdx.x	{	0----------------1----------------2----------------3----------------
	blockDim.x	{	                = 16             = 16             = 16             = 16
	threadIdx.x	{	-0123456789......-0123456789......-0123...........15...............15
 */
	int i = blockIdx.x*blockDim.x + threadIdx.x;						/*/*/	// i lets us know where in the array this kernel copy should target
	// be sure we aren't overstepping the array size- segfault!					// ... where it used to tell us where the loop iteration should target
	if (i<arraySize) {													/*/*/	// notice- no for loop! the looping action is totally parallelized
		sum[i] = add1[i] + add2[i];
	}
}

// all the action
int main(){

	// three arrays; we will add the first two to sum[]
	printf("initializing arrays\n");
	float *add1, *add2, *sum;
	cudaMallocManaged( &add1, arraySize*sizeof(float) );
	cudaMallocManaged( &add2, arraySize*sizeof(float) );
	cudaMallocManaged( &sum,  arraySize*sizeof(float) );

	// fill first two arrays before the CUDA starts
	for (int i=0; i<arraySize; i++){
		add1[i] = 1.0;
		add2[i] = 2.0;
	}
	printf("arrays done. prepare for adding\n");

	// parallelization happens here
	addArrays<<<4096,256>>>(arraySize, add1,add2,sum);					/*/*/	// look at all those kernels! 4096*256 = 1048576 = 1<<20

	// wait for all threads to complete on the GPU
	cudaDeviceSynchronize();
	printf("adding complete.\t");

	// check for accuracy- what's the biggest mistake?
	float maxError = 0.0;
	for (int i=0; i<arraySize; i++){
		// check each array index for value and store the greater deviation from 3.0
		maxError = fmax(maxError, fabs(sum[i]-3.0));
	}
	printf("max error = %f\n",maxError);

	// free memory
	cudaFree(add1);
	cudaFree(add2);
	cudaFree(sum);

	return 0;
}
