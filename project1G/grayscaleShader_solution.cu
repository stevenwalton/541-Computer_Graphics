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

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

/****************************************************/
/*                 Student Code                     */
/****************************************************/
__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    //int index = yIndex * numCols + xIndex;
    int index = yIndex * numRows + xIndex;
    if(index > (numCols*numRows)) return;
    uchar4 rgba = rgbaImage[index];

    unsigned char I = 0.299f*rgba.x + 0.587f*rgba.y + 0.114f*rgba.z;
    //unsigned char I = __fadd_rn(__fadd_rn(__fmul_rn(0.299f, rgba.x),
    //                  __fmul_rn(0.587f, rgba.y)),
    //                  __fmul_rn(0.114f, rgba.z));
    //greyImage[index] = I;
    greyImage[index] = static_cast<unsigned char>(I);
}

void 
your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, 
                       uchar4 * const d_rgbaImage,
                       unsigned char* const d_greyImage, 
                       size_t numRows, size_t numCols)
{
    int blockWidth = 32;
    const dim3 blockSize(blockWidth, blockWidth, 1);  //TODO
    const dim3 gridSize(numRows/blockWidth + 1, numCols/blockWidth + 1, 1);  //TODO
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}
/****************************************************/
/****************************************************/


cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

  //output the image
  cv::imwrite(output_file.c_str(), output);
}

void cleanup()
{
  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
  cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

  cv::imwrite(output_filename, reference);

}

template<typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
  //check that the GPU result matches the CPU result
  for (size_t i = 0; i < numElem; ++i) {
    if (ref[i] != gpu[i]) {
      std::cerr << "Difference at pos " << i << std::endl;
      //the + is magic to convert char to int without messing
      //with other types
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                 "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
  }
}

template<typename T>
void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
  assert(eps1 >= 0 && eps2 >= 0);
  unsigned long long totalDiff = 0;
  unsigned numSmallDifferences = 0;
  for (size_t i = 0; i < numElem; ++i) {
    //subtract smaller from larger in case of unsigned types
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > 0 && diff <= eps1) {
      numSmallDifferences++;
    }
    else if (diff > eps1) {
      std::cerr << "Difference at pos " << +i << " exceeds tolerance of " << eps1 << std::endl;
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
        "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
    totalDiff += diff * diff;
  }
  double percentSmallDifferences = (double)numSmallDifferences / (double)numElem;
  if (percentSmallDifferences > eps2) {
    std::cerr << "Total percentage of non-zero pixel difference between the two images exceeds " << 100.0 * eps2 << "%" << std::endl;
    std::cerr << "Percentage of non-zero pixel differences: " << 100.0 * percentSmallDifferences << "%" << std::endl;
    exit(1);
  }
}

//Uses the autodesk method of image comparison
//Note the the tolerance here is in PIXELS not a percentage of input pixels
template<typename T>
void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
{

  size_t numBadPixels = 0;
  for (size_t i = 0; i < numElem; ++i) {
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > variance)
      ++numBadPixels;
  }

  if (numBadPixels > tolerance) {
    std::cerr << "Too many bad pixels in the image." << numBadPixels << "/" << tolerance << std::endl;
    exit(1);
  }
}


struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};


void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows, size_t numCols)
{
  for (size_t r = 0; r < numRows; ++r) {
    for (size_t c = 0; c < numCols; ++c) {
      uchar4 rgba = rgbaImage[r * numCols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      //float channelSum = __mm_fmmul_ps(.299f,rgba.x) + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * numCols + c] = channelSum;
    }
  }
}


void
compareImages(std::string reference_filename, std::string test_filename,
              bool useEpsCheck, double perPixelError, double globalError)
{
    cv::Mat reference = cv::imread(reference_filename, -1);
    cv::Mat test = cv::imread(test_filename, -1);

    cv::Mat diff = abs(reference - test);

    cv::Mat diffSingleChannel = diff.reshape(1,0);

    double minVal, maxVal;

    cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL);

    diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));
    diff = diffSingleChannel.reshape(reference.channels(), 0);

    cv::imwrite("HW1_differenceImage.png", diff);

    unsigned char *referencePtr = reference.ptr<unsigned char>(0);
    unsigned char *testPtr = test.ptr<unsigned char>(0);

    if (useEpsCheck)
        checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols
                    * reference.channels(), perPixelError, globalError);
    else
        checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols
                    * reference.channels());
    std::cout << "PASS" << std::endl;
    return;
}



int main(int argc, char **argv)
{
    uchar4        *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_greyImage, *d_greyImage;
    std::string input_file;
    std::string output_file;
    std::string reference_file; // Don't use
    double perPixelError = 0.0;
    double globalError   = 0.0;
    bool useEpsCheck = false;

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
        case 6:
            useEpsCheck = true;
            input_file = std::string(argv[1]);
            output_file = std::string(argv[2]);
            reference_file = std::string(argv[3]);
            perPixelError = atof(argv[4]);
            globalError = atof(argv[5]);
            break;
        default:
            std::cerr << "Usage: ./HW1 input_file [output_filename]" 
                      << "[reference_filename] [perPixelError] [globalError]"
                      << std::endl;
            exit(1);
    }

    // load image and give input and output pointers
    preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
    GpuTimer timer;
    timer.Start();
    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
    timer.Stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    if (err < 0)
    {
        std::cerr << "Couldn't print timing info! STDOUT Closed!" << std::endl;
        exit(1);
    }

    size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * 
                numPixels, cudaMemcpyDeviceToHost));

    postProcess(output_file, h_greyImage);
    referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());
    postProcess(reference_file, h_greyImage);

    compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

    cleanup();

    return 0;
}
