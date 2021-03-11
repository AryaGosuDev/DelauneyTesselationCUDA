
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace cv;
cudaTextureObject_t texObjLinear;
constexpr bool CV_SUCCESS = true;
cudaArray* d_imageArray = 0;
size_t pitch;
uchar* d_img;

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

#define CHECK_CV(call) { \
    const bool error = call; \
    if (error != CV_SUCCESS) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason\n", error); \
        exit(-10 * error);\
    } \
} \

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
        ((unsigned int)(rgba.z * 255.0f) << 16) |
        ((unsigned int)(rgba.y * 255.0f) << 8) |
        ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c) {
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

__global__ void extractGradients(uchar* output, const cudaTextureObject_t texObj,  int width, int height) {

    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float u = x / (float)width;
    float v = y / (float)height;

    // Transform coordinates
    //u -= 0.5f;
    //v -= 0.5f;

    //B     G     R       stride

    output[y * (width * 3) + (x * 3)] = tex2D<uchar>(texObj, x * 3, y);
    output[y * (width * 3) + (x * 3) + 1] = tex2D<uchar>(texObj, x * 3 + 1, y);
    output[y * (width * 3) + (x * 3) + 2] = tex2D<uchar>(texObj, x * 3 + 2, y);
}

void returnGPUCudaInfoResources(int deviceID) {

    printf("Starting...\n");

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    checkCudaErrors(cudaSetDevice(deviceID));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
        driverVersion / 1000, (driverVersion % 100) / 10,
        runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
        deviceProp.major, deviceProp.minor);

    printf("  Total amount of global memory:                 %.2f GBytes (%llu bytes)\n",
        (float)deviceProp.totalGlobalMem / std::pow(1024.0, 3), (unsigned long long)deviceProp.totalGlobalMem);
    
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
        deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
        deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
            deviceProp.l2CacheSize);
    }

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
        "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
        deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
        deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
        "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
        deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
        deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n",
        deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
        deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
        deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
        deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
        deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
        deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n",
        deviceProp.memPitch);  
}

void initTexture(int w, int h, cv::Mat& _img) {

    checkCudaErrors(cudaMallocPitch(&d_img, &pitch, _img.step1(), h));
    checkCudaErrors(cudaMemcpy2D(d_img, pitch, _img.data, _img.step1(),
        w * 3 * sizeof(uchar), h, cudaMemcpyHostToDevice));
    
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = d_img;
    texRes.res.pitch2D.desc = desc;
    texRes.res.pitch2D.width = w*3;
    texRes.res.pitch2D.height = h;
    texRes.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc         texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObjLinear, &texRes, &texDescr, NULL));
}

int main() {
    try {

        returnGPUCudaInfoResources(0);
        const std::string filename = ".\\standard_test_images\\standard_test_images\\lena_color_256.tif";
        cv::Mat image = imread(filename, IMREAD_COLOR);

        /*
        uchar* testOutImage;
        testOutImage = (uchar*)malloc(image.cols * image.rows * sizeof(uchar) * 3);
        memcpy(testOutImage, image.data, image.cols * image.rows * sizeof(uchar) * 3);
        cv::Mat imageOutTest = cv::Mat(image.rows, image.cols, CV_8UC3, testOutImage);
        CHECK_CV(imwrite(".\\standard_test_images\\standard_test_images\\lena_color_256_test.png", imageOutTest));
        */

        if (image.empty()) {
            printf("Cannot read image file: %s\n", filename.c_str());
            return -1;
        }

        Point pt; pt.x = 10; pt.y = 8;
        cv::circle(image, pt, 2, 1);

        initTexture(image.cols, image.rows, image);
       
        // Allocate result of transformation in device memory
        uchar* d_output;
        checkCudaErrors(cudaMalloc((void **) &d_output, image.cols * image.rows * sizeof(uchar) * 3));

        uchar* gpuRef;
        gpuRef = (uchar*)malloc(image.cols * image.rows * sizeof(uchar) * 3);

        // Invoke kernel
        dim3 dimBlock(16, 16);
        dim3 dimGrid((image.cols + dimBlock.x - 1) / dimBlock.x,
            (image.rows + dimBlock.y - 1) / dimBlock.y);
        printf("Kernel Dimension :\n   Block size : %i , %i \n    Grid size : %i , %i",
            dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        extractGradients <<< dimGrid, dimBlock >>> (d_output, texObjLinear, image.cols, image.rows);

        checkCudaErrors(cudaMemcpy(gpuRef, d_output, image.cols * image.rows * sizeof(uchar) * 3, cudaMemcpyDeviceToHost));
        cv::Mat imageOut = cv::Mat(image.rows, image.cols, CV_8UC3, gpuRef);
        CHECK_CV(imwrite(".\\standard_test_images\\standard_test_images\\lena_color_256.png", imageOut));

        if (CV_SUCCESS == true) printf("\nSuccess !");

    }
    catch (Exception ex) {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}