/*	this is the tutorial written by Andrew Jones on CUDA
 *	the end goal is to add two very large arrays of floats together into a third array.
 *	the document is a series of code snippets which progress through learning CUDA.
 *	based on the tutorial:
https://devblogs.nvidia.com/even-easier-introduction-cuda/
 *	all changes from document to document are marked with this...*/				/*/*/	// changes between snippets look like this


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//

/*	let's start with basic C++
 *	this snippet is compiled with:
		g++ cuda0.cpp -o cuda0
*/
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//

#include <cstdio>
#include <cmath>

int arraySize = 1<<20; // basically a million

// function to add them together
void addArrays (int arraySize, float *add1, float *add2, float *sum){
	for (int i=0; i<arraySize; i++){
		sum[i] = add1[i] + add2[i];
	}
}

// all the action
int main(){

	// three arrays; we will add the first two to sum[]
	printf("initializing arrays\n");
	float *add1 = new float[arraySize];
	float *add2 = new float[arraySize];
	float *sum = new float[arraySize];

	// fill first two arrays before the CUDA starts
	for (int i=0; i<arraySize; i++){
		add1[i] = 1.0;
		add2[i] = 2.0;
	}
	printf("arrays done. prepare for adding\n");

	// CUDA will go here eventually
	addArrays(arraySize, add1,add2,sum);

	printf("adding complete.\t");

	// check for accuracy- what's the biggest mistake?
	float maxError = 0.0;
	for (int i=0; i<arraySize; i++){
		// check each array index for value and store the greater deviation from 3.0
		maxError = fmax(maxError, fabs(sum[i]-3.0));
	}
	printf("max error = %f\n",maxError);

	// free memory
	delete [] add1;
	delete [] add2;
	delete [] sum;

	return 0;
}
