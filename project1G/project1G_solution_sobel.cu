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

__global__ void
sobel_filter(unsigned char* const greyImage,
             unsigned char* const sobel_out,
             int numRows, int numCols)
{
    extern __shared__ uchar s_input2[];

    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    int index = yIndex*numRows + xIndex;
    if(index > (numCols*numRows)) return;
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
    for(int i = -r; i <= r; ++i)
    {
        int crtY = threadIdx.y + i;
        if(crtY < 0) crtY = 0;
        else if (crtY >= blockDim.y) crtY = blockDim.y -1;

        for(int j = -r; j <= r; ++j)
        {
            int crtX = threadIdx.x + j;
            if(crtX < 0) crtX = 0;
            else if(crtX >= blockDim.x) crtX = blockDim.x -1;

            const float inputPix = (float)s_input2[crtY*blockDim.x + crtX];
            sum_x += inputPix * dx[r+j][r+i];
            sum_y += inputPix * dy[r+j][r+i];
        }
    }
    sobel_out[index] = (uchar)(abs(sum_x)+abs(sum_y));
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

void
your_sobel(unsigned char* const h_greyImage,
           unsigned char* const d_greyImage,
           unsigned char* const d_sobel,
           size_t numRows, size_t numCols)
{
    const dim3 blockDimHist(16,16,1);
    const dim3 gridDimHist(ceil((float)numCols/blockDimHist.x), ceil((float)numRows/blockDimHist.y),1);
    size_t blockSharedMemory = blockDimHist.x*blockDimHist.y*sizeof(uchar);
    sobel_filter<<<gridDimHist, blockDimHist, blockSharedMemory>>>(d_greyImage, d_sobel, numRows, numCols);

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

    cudaMalloc((void**)&d_rgbaImage, sizeof(uchar4)*numPixels);
    cudaMalloc((void**)&d_greyImage, sizeof(unsigned char) * numPixels);
    cudaMalloc((void**)&d_sobel, sizeof(unsigned char)*numPixels);
    cudaMemset(d_greyImage, 0, numPixels * sizeof(unsigned char));
    cudaMemset(d_sobel, 0, numPixels * sizeof(unsigned char));

    cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel, h_sobel, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice);


    //your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, d_sobel, numRows, numCols);
    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char)*numPixels, cudaMemcpyDeviceToHost);
    // Write out greyscale
    cv::Mat output(numRows, numCols, CV_8UC1, (void*)h_greyImage);
    cv::imwrite(greyscale_file.c_str(), output);

    cudaDeviceSynchronize();
    // Now we'll do the Sobel filter
    your_sobel(h_greyImage, d_greyImage, d_sobel, numRows, numCols);
    cudaMemcpy(h_sobel, d_sobel, sizeof(unsigned char)*numPixels, cudaMemcpyDeviceToHost);

    // Write Sobel
    cv::Mat output2(numRows, numCols, CV_8UC1, (void*)h_sobel);
    cv::imwrite(sobel_file.c_str(), output2);

    // Free \o/
    cudaFree(d_rgbaImage);
    cudaFree(d_greyImage);
    cudaFree(d_sobel);
    return 0;
}
