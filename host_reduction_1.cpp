#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>


const char* programSource =
"__kernel void reduction_total(__global int* bufA,\n"
"	__local int* bufB, __global int* bufC) {\n"
"\n"
"	int lid = get_local_id(0);\n"
"	int group_size = get_local_size(0);\n"
"\n"
"	bufB[lid] = bufA[get_global_id(0)];\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"for (int i = group_size / 2; i > 0; i /= 2) {\n"
"	if (lid < i)\n"
"		bufB[lid] += bufB[lid + i];\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"}\n"
"	if(lid==0)\n"
"	bufC[get_group_id(0)] = bufB[0];\n"
"}\n"
;

//const char* programSource_2 =



int main() {
	const int elements = 256;
	size_t datasize = sizeof(int)*elements;

	int *A = (int*)malloc(datasize);
	int *B = (int*)malloc(datasize);
	int *C = (int*)malloc(sizeof(int) * 16);
	for (int i = 0; i < elements; i++) {
		A[i] = 1;
	}

	cl_int status; // used for error checking.
	cl_platform_id platform;
	cl_device_id device;

	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(1, NULL, &numPlatforms);
	status = clGetPlatformIDs(1, &platform, NULL);

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);


	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);


	cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &status);
	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	//cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
	cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 16 * sizeof(int), NULL, &status);

	status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);



	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);

	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	status = clBuildProgram(program, 0, &device, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "reduction_total", &status);


	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA); //__global
	status = clSetKernelArg(kernel, 1, sizeof(int)*16, NULL); //__local
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC); //__global




	size_t indexSpaceSize[1], workGroupSize[1];

	indexSpaceSize[0] = elements;
	workGroupSize[0] = 16;


	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
	status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, 16*sizeof(int), C, 0, NULL, NULL);



	for (int i = 0; i < 16; i++)
		printf("C[%d] == %d\n", i, C[i]);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufA);

	clReleaseMemObject(bufC);
	clReleaseContext(context);

	free(A);
	free(C);


	system("pause");
	return 0;

}