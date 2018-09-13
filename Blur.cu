#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <strings.h>
#include "common.h"

#define MS 5;

using namespace std;
using namespace cv;

__global__ void Blur_Kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int ref){

    // 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    
    
    if ((xIndex < width) && (yIndex < height))
	{
        int blue = 0; 
        int green = 0; 
        int red = 0;

        int contador=0;

        const int color_tid = (yIndex) * colorWidthStep + (3 * xIndex);

        for(int i = -ref ; i <= ref ; i++){
            for (int j = -ref; j<=ref ; j++){

                //Location of colored pixel in input
                const int color_tid = (yIndex+i) * colorWidthStep + (3 * (xIndex+j));
                if(xIndex+j>0 && yIndex+i>0 && xIndex+j<width && yIndex+i<height ){
                
                    contador++;

                    blue += input[color_tid];
                    green += input[color_tid+1];
                    red += input[color_tid+2];

                }
            }
        }

        output[color_tid] = static_cast<unsigned char>(blue/contador);
		output[color_tid+1] = static_cast<unsigned char>(green/contador);
        output[color_tid+2] = static_cast<unsigned char>(red/contador);   
	}
}

void Blur(string file){

    int ms = MS;
    int ref = floor(ms/2);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    Mat input = cv::imread(file, CV_LOAD_IMAGE_COLOR);
    cout << "Input image step: " << input.step << " cols: " << input.cols << " rows: " << input.rows << endl;

    //Create output image
    Mat output(input.rows, input.cols, CV_8UC3);
    

    // Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;
    size_t grayBytes = output.step * output.rows;
    
    unsigned char *d_input, *d_output;

    // Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_output, output.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
    const dim3 block(16, 16);
    
    // Calculate grid size to cover the whole image
    const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
    printf("Blur_Kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
    
    // Launch the color conversion kernel
    Blur_Kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), ref);
    
    // Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
    
    //Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
    imshow("Output", output);
    

}


int main(int argc, char *argv[]){

    if (argc < 2){
         
        cout << "No hay argumentos suficientes" << endl;

    }else{
        auto startTime = chrono::high_resolution_clock::now();
        Blur(argv[1]);
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = endTime - startTime;

        printf("Blur elapsed %f ms\n", duration_ms.count());

        waitKey(0);
    }

}   
