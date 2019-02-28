#include <iostream>
#include <string>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

__global__ void
rgba_to_greyscale(void)
{
    // Don't forget to check if the index is out of bounds
    // A simple `return` will break out for us

    // Suggest you use a static_cast when converting back to your
    // grey image's index
}

__global__ void
sobel_filter(void)
{
    extern __shared__ uchar s_input2[];

    int sobelWidth = 3;
    int dx[3][3] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int dy[3][3] = { 1, 2, 1,
                     0, 0, 0,
                    -1,-2,-1};
    int sum_x = 0;
    int sum_y = 0;
    const int r = (sobelWidth - 1)/2;

    s_input2[threadIdx.y *blockDim.x + threadIdx.x] = greyImage[index];
    __syncthreads();
    // Do the kernel operations here
    // Will require a double for loop
    // 

    // Set your output to = (uchar)(abs(sum_x)+abs(sum_y));
}

void
your_rgba_to_greyscale(void)
{
    const dim3 blockSize(void);
    const dim3 gridSize(void);
    rgba_to_greyscale<<<>>>(void);
    cudaDeviceSynchronize();
}

void
your_sobel(void)
{
    const dim3 blockDimHist(16,16,1);
    const dim3 gridDimHist(void);
    size_t blockSharedMemory = blockDimHist.x*blockDimHist.y*sizeof(uchar);
    sobel_filter<<<gridDimHist, blockDimHist, blockSharedMemory>>>(void);

}

int main(int argc, char **argv)
{
    uchar4        *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_greyImage, *d_greyImage;
    unsigned char *h_sobel, *d_sobel;
    std::string input_file;
    std::string greyscale_file;
    std::string sobel_file;

    switch(argc)
    {
        case 2:
            input_file = std::string(argv[1]);
            greyscale_file = "project1G_greyscale.png";
            sobel_file = "project1G_sobel.png";
            break;
        case 3:
            input_file = std::string(argv[1]);
            greyscale_file = std::string(argv[2]);
            sobel_file = "project1G_sobel.png";
            break;
        case 4:
            input_file = std::string(argv[1]);
            greyscale_file = std::string(argv[2]);
            sobel_file = std::string(argv[3]);
            break;
        default:
            std::cerr << "Usage: ./project1G input_file [greyscale_filename]" 
                      << "[sobel_filename]"
                      << std::endl;
            exit(1);
    }

    cv::Mat image;
    image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    if(image.empty())
    {
        std::cerr << "Couldn't open file: " << input_file << std::endl;
        exit(1);
    }
    cv::Mat imageRGBA;
    cv::Mat imageGrey;
    cv::Mat imageSobel;
    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

    imageGrey.create(image.rows, image.cols, CV_8UC1);
    imageSobel.create(image.rows, image.cols, CV_8UC1);

    if(!imageRGBA.isContinuous() || !imageGrey.isContinuous())
    {
        std::cerr << "Images aren't continuous. Exiting" << std::endl;
        exit(1);
    }

    *(&h_rgbaImage) = (uchar4*)imageRGBA.ptr<unsigned char>(0);
    *(&h_greyImage) = imageGrey.ptr<unsigned char>(0);
    *(&h_sobel) = imageSobel.ptr<unsigned char>(0);
    size_t numRows = imageRGBA.rows;
    size_t numCols = imageRGBA.cols;
    size_t numPixels = numRows * numCols;

    // Allocate all your memory here. Use cudaMallocs, Memset, and Memcpy



    your_rgba_to_greyscale(void);
    cudaDeviceSynchronize();

    // Do a memcpy for your grey Image here (Device to Host)
    cudaMemcpy(void);

    // Write out greyscale
    cv::Mat output(numRows, numCols, CV_8UC1, (void*)h_greyImage);
    cv::imwrite(greyscale_file.c_str(), output);

    cudaDeviceSynchronize();
    // Now we'll do the Sobel filter
    your_sobel(void);
    cudaMemcpy(void);

    // Write Sobel
    cv::Mat output2(numRows, numCols, CV_8UC1, (void*)h_sobel);
    cv::imwrite(sobel_file.c_str(), output2);

    // Free \o/
    cudaFree(d_rgbaImage);
    cudaFree(d_greyImage);
    cudaFree(d_sobel);
    return 0;
}
