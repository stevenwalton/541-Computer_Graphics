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

void
your_rgba_to_greyscale(void)
{
    // Fill in everything here
    const dim3 blockSize(void);
    const dim3 gridSize(void);
    rgba_to_greyscale<<<void>>>(void);
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
            output_file = "project1G_output.png";
            reference_file = "project1G_reference.png";
            break;
        case 3:
            input_file = std::string(argv[1]);
            output_file = std::string(argv[2]);
            reference_file = "project1G_reference.png";
            break;
        case 4:
            input_file = std::string(argv[1]);
            output_file = std::string(argv[2]);
            reference_file = std::string(argv[3]);
            break;
        default:
            std::cerr << "Usage: ./project1G input_file [output_filename]" 
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

    // Allocate your memory here. Use cudaMallocs, Memset, and Memcpy


    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize();

    // Copy back data to the host
    cv::Mat output(numRows, numCols, CV_8UC1, (void*)h_greyImage);
    cv::imwrite(output_file.c_str(), output);

    // Don't forget to free your memory

    return 0;
}
