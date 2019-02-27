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
rgba_to_greyscale(const uchar4* const rgbaImage,
                  unsigned char* const greyImage,
                  int numRows, int numCols)
{
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;

    int index = yIndex*numRows + xIndex;
    if(index > (numCols*numRows)) return;
    uchar4 rgba = rgbaImage[index];

    unsigned char I = 0.299f*rgba.x + 0.587f*rgba.y + 0.114f*rgba.z;

    greyImage[index] = static_cast<unsigned char>(I);
}

void
your_rgba_to_greyscale(const uchar4* const h_rgbaImage,
                       uchar4* const d_rgbaImage,
                       unsigned char* const d_greyImage,
                       size_t numRows, size_t numCols)
{
    int blockWidth = 32;
    const dim3 blockSize(blockWidth, blockWidth, 1);
    const dim3 gridSize(numRows/blockWidth + 1, numCols/blockWidth + 1,1);
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    uchar4        *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_greyImage, *d_greyImage;
    std::string input_file;
    std::string output_file;
    std::string reference_file;

    switch(argc)
    {
        case 2:
            input_file = std::string(argv[1]);
            output_file = "HW1_output.png";
            reference_file = "HW1_reference.png";
            break;
        case 3:
            input_file = std::string(argv[1]);
            output_file = std::string(argv[2]);
            reference_file = "HW1_reference.png";
            break;
        case 4:
            input_file = std::string(argv[1]);
            output_file = std::string(argv[2]);
            reference_file = std::string(argv[3]);
            break;
        default:
            std::cerr << "Usage: ./HW1 input_file [output_filename]" 
                      << "[reference_filename] [perPixelError] [globalError]"
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
    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

    imageGrey.create(image.rows, image.cols, CV_8UC1);

    if(!imageRGBA.isContinuous() || !imageGrey.isContinuous())
    {
        std::cerr << "Images aren't continuous. Exiting" << std::endl;
        exit(1);
    }

    *(&h_rgbaImage) = (uchar4*)imageRGBA.ptr<unsigned char>(0);
    *(&h_greyImage) = imageGrey.ptr<unsigned char>(0);
    size_t numRows = imageRGBA.rows;
    size_t numCols = imageRGBA.cols;
    size_t numPixels = numRows * numCols;

    cudaMalloc((void**)&d_rgbaImage, sizeof(uchar4)*numPixels);
    cudaMalloc((void**)&d_greyImage, sizeof(unsigned char) * numPixels);
    cudaMemset(d_greyImage, 0, numPixels * sizeof(unsigned char));

    cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);


    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char)*numPixels, cudaMemcpyDeviceToHost);
    cv::Mat output(numRows, numCols, CV_8UC1, (void*)h_greyImage);
    cv::imwrite(output_file.c_str(), output);

    cudaFree(d_rgbaImage);
    cudaFree(d_greyImage);
    return 0;
}
