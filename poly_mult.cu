#include <cstdio>
#include <cassert>
#include <iostream>
#include <string>

/************************* NOTES *****************************

- The two input polynomials must have same degree, namely n-1
- The integer n must be a power of 2

**************************************************************/ 

using namespace std;

/*********** CUDA Helper functions from examples *******************/

struct cuda_exception {
    explicit cuda_exception(const char *err) : error_info(err) {}
    explicit cuda_exception(const string &err) : error_info(err) {}
    string what() const throw() { return error_info; }

    private:
    string error_info;
};

void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        string error_info(msg);
        error_info += " : ";
        error_info += cudaGetErrorString(err);
        throw cuda_exception(error_info);
    }
}

/*******************************************************************/

__global__ void poly_mult_ker(int *M, int n, int p) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = id % n;
    int j = id / n;
    int d = i + j;
    M[(2 * n) + (d * n) + i] = (M[i] * M[n + j]) % p;
}

__global__ void reduce_terms_ker(int *M, int n, int p) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 1; k < n; k *= 2) {
		if ( id % (2*k) == 0)
			M[(2*n) + id] = (M[(2*n) + id] + M[(2*n) + id + k]) % p;
		// Sync to ensure values are ready for this step
		__syncthreads();
	}

	// Sync to ensure the result is ready
	__syncthreads();
	if ( id % n == 0)
		M[(2*n) + (2*n - 1) * n + (id / n)] = (M[(2*n) + (2*n - 1) * n + (id / n)] + M[(2*n) + id]) % p;

	for (int k = 1; k < n; k *= 2) {
		if ( id % (2*k) == 0 && id < ( n * (n-1)))
			M[(2*n) + (n*n) + id] = (M[(2*n) + (n*n) + id] + M[(2*n) + (n*n) + id + k]) % p;
		// Sync to ensure values are ready for this step
		__syncthreads();
	}

	// Sync to ensure the result is ready
	__syncthreads();
	if ( id % n == 0 && id < ( n * (n-1)) )
		M[(2*n) + (2*n - 1) * n + n + (id / n)] = (M[(2*n) + (2*n - 1) * n + n + (id / n)] + M[(2*n) + (n*n) + id]) % p;

}

__global__ void reduce_terms_ker_q2(int *M, int n, int p) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 1; k < n; k *= 2) {
		if ( id % (2*k) == 0)
			M[(2*n) + id] = (M[(2*n) + id] + M[(2*n) + id + k]) % p;
		// Sync to ensure values are ready for this step
		__syncthreads();
	}
	// Sync to ensure the result is ready
	__syncthreads();
	if ( id % n == 0)
		M[(2*n) + (2*n - 1) * n + (id / n)] = (M[(2*n) + (2*n - 1) * n + (id / n)] + M[(2*n) + id]) % p;

	for (int k = 1; k < n; k *= 2) {
		if ( id % (2*k) == 0 && id < ( n * (n-1)))
			M[(2*n) + (n*n) + id] = (M[(2*n) + (n*n) + id] + M[(2*n) + (n*n) + id + k]) % p;
		// Sync to ensure values are ready for this step
		__syncthreads();
	}

	// Sync to ensure the result is ready
	__syncthreads();
	if ( id % n == 0 && id < ( n * (n-1)) )
		M[(2*n) + (2*n - 1) * n + n + (id / n)] = (M[(2*n) + (2*n - 1) * n + n + (id / n)] + M[(2*n) + (n*n) + id]) % p;
}

void random_polynomials(int *M, size_t n, int p) {
	for ( int i = 0; i < 2*n; i++ ){
		int num = (int) rand() % p;
		M[i] = num;
    }
}

// Run kernel with coefficients of all 1 for simple verification
int poly_mult_test(int n_terms, int modulo_p, int question_id, int n_b, int n_t) {

    const int n = n_terms;
    const int p = modulo_p;

    // size n for a, size n for b, size (2*n-1) * n for coefficients of each term, size 2*n-1 for final summed coefficients
    const int worksp_size = (2 * n + ((2 * n - 1) * n) + (2*n)-1 );

    // Set coefficients to be 1 to easily verify the result
    int M[worksp_size] = {0};
    for (int i = 0; i < 2*n; i++) {
    	M[i] = 1;
    }

    // Display input polynomials
    printf("\n============== INPUT & RESULTS ==========\n\n");
    printf("Input - Polynomial A Coefficients:\n");
    for (int i=0; i<n; i++) {
		printf("%d ", M[i]);
	}
	printf("\n\nInput - Polynomial B Coefficients:\n");
	for (int i=0; i<n; i++) {
		printf("%d ", M[i+n]);
	}
	printf("\n\n");

	// Allocate GPU memory for the workspace
    int *Md;
    cudaMalloc((void **)&Md, sizeof(int)*worksp_size);
    checkCudaError("allocate GPU memory for the workspace");
    cudaMemcpy(Md, M, sizeof(int)*worksp_size, cudaMemcpyHostToDevice);

    poly_mult_ker<<<n_b, n_t>>>(Md, n, p);

    if (question_id == 1) {
	    reduce_terms_ker<<<n_b, n_t>>>(Md, n, p);
	} else if (question_id == 2) {
	    reduce_terms_ker_q2<<<n_b, n_t>>>(Md, n, p);
	}

	// Copy GPU memory for the workspace back to host
    cudaMemcpy(M, Md, sizeof(int)*worksp_size, cudaMemcpyDeviceToHost);	

    // Display resulting polynomial
    int result_start = 2 * n + ((2 * n - 1) * n);
    int result_length = (2 * n) - 1;
    printf("Result - Polynomial A*B Coefficients:\n");
    for (int i = result_start; i < result_start + result_length; i++)
    	printf("%d ", M[i]);
    printf("\n");

    int isCorrect = 1;
    for (int i = result_start; i < result_start + result_length - 1; i++) {
    	int diff = abs(M[i] - M[i+1]);
    	if ( !(diff == 1 || diff == (p - 1)) ) isCorrect = 0;
    }

    if (isCorrect) {
    	printf("\nResult is correct!\n");
    } else {
    	printf("\nResult is INCORRECT!\n");
    }

    cudaFree(Md);

    return 0;
}

// Run kernel with random polynomial input
int poly_mult(int n_terms, int modulo_p, int q, int n_b, int n_t, int is_dev_mode) {
    const int n = n_terms;
    const int p = modulo_p;

    // size n for a, size n for b, size (2*n-1) * n for coefficients of each term, size 2*n-1 for final summed coefficients
    const int worksp_size = (2 * n + ((2 * n - 1) * n) + (2*n)-1 );
    int M[worksp_size] = {0};
    random_polynomials(M, n, p);

    // Display input polynomials
    printf("\n============== INPUT & RESULTS ==========\n\n");
    printf("Input - Polynomial A Coefficients:\n");
    for (int i=0; i<n; i++) {
		printf("%d ", M[i]);
	}
	printf("\n\nInput - Polynomial B Coefficients:\n");
	for (int i=0; i<n; i++) {
		printf("%d ", M[i+n]);
	}
	printf("\n\n");

	// Allocate GPU memory for the workspace
    int *Md;
    cudaMalloc((void **)&Md, sizeof(int)*worksp_size);
    checkCudaError("allocate GPU memory for the workspace");
    cudaMemcpy(Md, M, sizeof(int)*worksp_size, cudaMemcpyHostToDevice);

    poly_mult_ker<<<n_b, n_t>>>(Md, n, p);

    if (q == 1) {
	    reduce_terms_ker<<<n_b, n_t>>>(Md, n, p);
	} else if (q == 2) {
	    reduce_terms_ker_q2<<<n_b, n_t>>>(Md, n, p);
	}

	// Copy GPU memory for the workspace back to host
    cudaMemcpy(M, Md, sizeof(int)*worksp_size, cudaMemcpyDeviceToHost);

    // Display workspace values if dev mode is true
    if (is_dev_mode) {
	    //Debug workspace
	    printf("\nDEBUG WORKSPACE:\n");
	    for (int i = 2*n; i < worksp_size; ++i) {
	    	if (i % n == 0)
	    		printf(" . ");
	    	printf("%d ", M[i]);
	    }
	    printf("\n\n\n");	
    }

    // Display resulting polynomial
    int result_start = 2 * n + ((2 * n - 1) * n);
    int result_length = (2 * n) - 1;
    printf("Result - Polynomial A*B Coefficients:\n");
    for (int i = result_start; i < result_start + result_length; i++)
    	printf("%d ", M[i]);
    printf("\n");

    cudaFree(Md);

    return 0;
}

void print_usage() {
	printf("Usage: ./poly_mult [MODE: run, dev, test] [QUESTION ID: (integer) 1 OR 2] [ N_TERMS: integer power of 2 ] [ MODULO_P: integer small prime (e.g. 103) ]\n");
}

int validate_args(int argc, char **argv) {
	if ( argc < 3 )
		return 1;

	string mode = argv[1];
	int question_id = atoi(argv[2]);

	if ( !(mode.compare("run") == 0) && !(mode.compare("dev") == 0) && !(mode.compare("test") == 0) ) {
		printf("Invalid Mode!\n");
		return 1;
	}

	if ( ! (question_id == 1 || question_id == 2) ) {
		printf("Invalid Question Id!\n");
		return 1;
	}

	if ( argc < 5 ) {
		printf("Please provide values for both: [N_TERMS] [MODULO_P]\n");
		return 1;
	}

	return 0;
}

int main(int argc, char **argv) {

	if (validate_args(argc, argv)) {
		print_usage();
		exit(1);
	}

	string mode = argv[1];
	int question_id = atoi(argv[2]);
	int n_terms = atoi(argv[3]);
	int modulo_p = atoi(argv[4]);
	int n_blocks, n_threads;

	if (question_id == 1) {
		n_threads = n_terms;
		n_blocks = n_terms;
	} else if (question_id == 2) {
		n_threads = 64;
		n_blocks = (n_terms*n_terms) / n_threads;
	}

	printf("\n======= ARGUMENTS =======\n");
	cout << "MODE: " << argv[1] << "\n";
	printf("question_id: %d\n", question_id);
	printf("n_terms: %d\n", n_terms);
	printf("modulo_p: %d\n", modulo_p);
	printf("n_blocks: %d\n", n_blocks);
	printf("n_threads: %d\n", n_threads);
	printf("=========================\n");


	if ( mode.compare("run") == 0 ) {
		poly_mult(n_terms, modulo_p, question_id, n_blocks, n_threads, 0);
	}

	if ( mode.compare("dev") == 0 ) {
		poly_mult(n_terms, modulo_p, question_id, n_blocks, n_threads, 1);
	}

	if ( mode.compare("test") == 0 ) {
		poly_mult_test(n_terms, modulo_p, question_id, n_blocks, n_threads);
	}

	return 0;
}