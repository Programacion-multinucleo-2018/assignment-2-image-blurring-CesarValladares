#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include "omp.h"

#define MS 5;

using namespace std;
using namespace cv;

void BlurImage(const Mat &input, Mat &output, int cols, int rows){

    int ms = MS;
    int ref = floor(ms/2);

    for (int i = 0 ; i < cols ; i ++){
        for (int j = 0 ; j < rows ; j++){

            int blue = 0;
            int green = 0;
            int red = 0;

            int contador = 0;

            for (int ii = -ref ; ii <= ref ; ii++){
                for (int jj = - ref ; jj <=ref ; jj++){

                    if ( i + ii < 0 && j + jj < 0 && j + jj > rows && i + ii > cols ){

                        //Casilla de la matriz aux fuera de la imagen
                    }
                    else
                    {
                        contador++;

                        blue += input.at<Vec3b>(j+jj, i+ii)[0];
                        green += input.at<Vec3b>(j+jj, i+ii)[1];
                        red += input.at<Vec3b>(j+jj, i+ii)[2];

                    }

                }
            }

            output.at<Vec3b>(j,i)[0] = (blue/contador);
            output.at<Vec3b>(j,i)[1] = (green/contador);
            output.at<Vec3b>(j,i)[2] = (red/contador);            

        }
    }
}

void Blur(string file){

    Mat image;

    image = imread(file, CV_LOAD_IMAGE_COLOR);

    if(!image.data)
    {
        cout <<  "Could not open or find the image" << std::endl ;
        
    }else{

        int x = image.cols;
        int y = image.rows;

        Mat output(image.rows, image.cols, CV_8UC3);

        cout << "Input image step: " << image.step << " cols: " << x << " rows: " << y << endl;


        BlurImage(image, output, x, y);

        namedWindow("Original", cv::WINDOW_NORMAL);
        namedWindow("Output", cv::WINDOW_NORMAL);

        imshow("Original", image);
        imshow("Output", output);
    }

}

int main (int argc, char** argv){

    

    if (argc < 2){
         
        cout << "No hay argumentos suficientes" << endl;

    }else{

        auto startTime = chrono::high_resolution_clock::now();
        Blur(argv[1]);
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = endTime - startTime;

        printf("Tiempo transcurrido:  %f ms \n", duration_ms.count());

    }

    waitKey(0); 

    return 0;
}


