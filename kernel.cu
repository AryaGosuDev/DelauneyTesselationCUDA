
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
cudaSurfaceObject_t  outputSurfRef;
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

__device__ void gaussianBlur(float* d_intensity_img, cudaSurfaceObject_t  outputSurfRef, const int x, const int y, const int width) {
    float tempGaussianValues[25]; float data;
    surf2Dread(&data, outputSurfRef, ((x - 2) * 4), y - 2, cudaBoundaryModeClamp); tempGaussianValues[0] = data;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y - 2, cudaBoundaryModeClamp); tempGaussianValues[1] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x) * 4),     y - 2, cudaBoundaryModeClamp); tempGaussianValues[2] = data * 7.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y - 2, cudaBoundaryModeClamp); tempGaussianValues[3] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x + 2) * 4), y - 2, cudaBoundaryModeClamp); tempGaussianValues[4] = data;
    surf2Dread(&data, outputSurfRef, ((x - 2) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[5] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[6] = data * 16.0f;
    surf2Dread(&data, outputSurfRef, ((x) * 4),     y - 1, cudaBoundaryModeClamp); tempGaussianValues[7] = data * 26.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[8] = data * 16.0f;
    surf2Dread(&data, outputSurfRef, ((x + 2) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[9] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x - 2) * 4), y    , cudaBoundaryModeClamp); tempGaussianValues[10] = data * 7.0f;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y    , cudaBoundaryModeClamp); tempGaussianValues[11] = data * 26.0f;
    surf2Dread(&data, outputSurfRef, ((x) * 4),     y    , cudaBoundaryModeClamp); tempGaussianValues[12] = data * 41.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y    , cudaBoundaryModeClamp); tempGaussianValues[13] = data * 26.0f;
    surf2Dread(&data, outputSurfRef, ((x + 2) * 4), y    , cudaBoundaryModeClamp); tempGaussianValues[14] = data * 7.0f;
    surf2Dread(&data, outputSurfRef, ((x - 2) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[15] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[16] = data * 16.0f;
    surf2Dread(&data, outputSurfRef, ((x) * 4),     y + 1, cudaBoundaryModeClamp); tempGaussianValues[17] = data * 26.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[18] = data * 16.0f;
    surf2Dread(&data, outputSurfRef, ((x + 2) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[19] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x - 2) * 4), y + 2, cudaBoundaryModeClamp); tempGaussianValues[20] = data;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y + 2, cudaBoundaryModeClamp); tempGaussianValues[21] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x) * 4),     y + 2, cudaBoundaryModeClamp); tempGaussianValues[22] = data * 7.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y + 2, cudaBoundaryModeClamp); tempGaussianValues[23] = data * 4.0f;
    surf2Dread(&data, outputSurfRef, ((x + 2) * 4), y + 2, cudaBoundaryModeClamp); tempGaussianValues[24] = data;
    data = 0.0f;
    for (int i = 0; i < 25; ++i) {
        data += tempGaussianValues[i];
    }
    data /= 273.0f;
    d_intensity_img[y * width + x] = data;
}

__device__ void gaussianBlurSmall(float* d_intensity_img, cudaSurfaceObject_t  outputSurfRef, const int x, const int y, const int width) {
    float tempGaussianValues[9]; float data;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[0] = data;
    surf2Dread(&data, outputSurfRef, ((x) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[1] = data * 2.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y - 1, cudaBoundaryModeClamp); tempGaussianValues[2] = data ;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y , cudaBoundaryModeClamp); tempGaussianValues[3] = data * 2.0f;
    surf2Dread(&data, outputSurfRef, ((x) * 4), y , cudaBoundaryModeClamp); tempGaussianValues[4] = data * 10.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y , cudaBoundaryModeClamp); tempGaussianValues[5] = data * 2.0f;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[6] = data;
    surf2Dread(&data, outputSurfRef, ((x) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[7] = data * 2.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y + 1, cudaBoundaryModeClamp); tempGaussianValues[8] = data;
    data = 0.0f;
    for (int i = 0; i < 9; ++i) {
        data += tempGaussianValues[i];
    }
    data /= 22.0f;
    d_intensity_img[y * width + x] = data;
}

__global__ void loadIntensitySurface (const cudaTextureObject_t texObj, cudaSurfaceObject_t  outputSurfRef, 
    float* d_intensity_img, int width, int height) {
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    d_intensity_img[y * width + x] = ((tex2D<uchar>(texObj, x * 3, y) +
                                       tex2D<uchar>(texObj, x * 3 + 1, y) +
                                       tex2D<uchar>(texObj, x * 3 + 2, y)) / 3.0f) / 255.0f;

    //write into the surface
    surf2Dwrite(d_intensity_img[y * width + x], outputSurfRef, (x * 4), y);

}

__global__ void extractGradients(uchar* output, float * output_float,
    float * x_grad, float * y_grad, float * d_intensity_img,
    const cudaTextureObject_t texObj, cudaSurfaceObject_t  outputSurfRef,  
    int width, int height) {

    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    

    //perform gaussian blur to diminish noise in reading gradient
    //gaussianBlurSmall(d_intensity_img, outputSurfRef, x, y, width);
    //surf2Dwrite(d_intensity_img[y * width + x], outputSurfRef, (x * 4), y);

    __syncthreads();

    float tempSobelValues[6]; float data; 
    surf2Dread(&data, outputSurfRef, ((x-1) * 4), y - 1, cudaBoundaryModeClamp); tempSobelValues[0] = data * 5.0f;
    surf2Dread(&data, outputSurfRef, ((x-1) * 4), y, cudaBoundaryModeClamp); tempSobelValues[1] = data * 8.0f;
    surf2Dread(&data, outputSurfRef, ((x-1) * 4), y+1, cudaBoundaryModeClamp); tempSobelValues[2] = data * 5.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y-1, cudaBoundaryModeClamp); tempSobelValues[3] = data * 5.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y, cudaBoundaryModeClamp); tempSobelValues[4] = data * 8.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y+1, cudaBoundaryModeClamp); tempSobelValues[5] = data * 5.0f;

    data = (tempSobelValues[0] + tempSobelValues[1] + tempSobelValues[2] - tempSobelValues[3] -
        tempSobelValues[4] - tempSobelValues[5]) / 36.0f;
    
    x_grad[y * width + x] = data;


    __syncthreads();

    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y - 1, cudaBoundaryModeClamp); tempSobelValues[0] = data * 5.0f;
    surf2Dread(&data, outputSurfRef, (x * 4), y - 1, cudaBoundaryModeClamp); tempSobelValues[1] = data * 8.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y - 1, cudaBoundaryModeClamp); tempSobelValues[2] = data * 5.0f;
    surf2Dread(&data, outputSurfRef, ((x - 1) * 4), y + 1, cudaBoundaryModeClamp); tempSobelValues[3] = data * 5.0f;
    surf2Dread(&data, outputSurfRef, (x * 4), y + 1, cudaBoundaryModeClamp); tempSobelValues[4] = data * 8.0f;
    surf2Dread(&data, outputSurfRef, ((x + 1) * 4), y + 1, cudaBoundaryModeClamp); tempSobelValues[5] = data * 5.0f;

    data = (tempSobelValues[0] + tempSobelValues[1] + tempSobelValues[2] - tempSobelValues[3] -
        tempSobelValues[4] - tempSobelValues[5]) / 36.0f;

    y_grad[y * width + x] = data;


    __syncthreads();

    float Ix2 = x_grad[y * width + x] * x_grad[y * width + x];
    float Iy2 = y_grad[y * width + x] * y_grad[y * width + x];
    float Ixy = x_grad[y * width + x] * y_grad[y * width + x];

    float eigen1 = ((Ix2 + Iy2) + sqrt((Ix2 * Ix2 + 2.0f * Ix2 * Iy2 + Iy2 * Iy2) - ( 4.0f * ( Ix2 * Iy2 - Ixy * Ixy)))) / 2.0f;
    float eigen2 = ((Ix2 + Iy2) - sqrt((Ix2 * Ix2 + 2.0f * Ix2 * Iy2 + Iy2 * Iy2) - ( 4.0f * ( Ix2 * Iy2 - Ixy * Ixy)))) / 2.0f;

    float R = eigen1 * eigen2 - ( (eigen1 + eigen2) * (eigen1 + eigen2));

    __syncthreads();
    
    //float data;
    surf2Dread(&data, outputSurfRef, (x * 4) ,y);
    //printf("%f\n", data);
    //float fillData = sqrt(y_grad[y * width + x] * y_grad[y * width + x] + x_grad[y * width + x] * x_grad[y * width + x]) * 255.0f;
    float fillData = abs(R) * 255.0f;
    output[y * (width * 3) + (x * 3)] = static_cast<uchar>(fillData);
    output[y * (width * 3) + (x * 3) + 1] = static_cast<uchar>(fillData);
    output[y * (width * 3) + (x * 3) + 2] = static_cast<uchar>(fillData);

    output_float[y * (width) + (x)] = abs(R);
    
    
    
    
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

void initSurface(int w, int h) {
    
    cudaArray* cuOutputArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&cuOutputArray, &channelDesc, w, h, cudaArraySurfaceLoadStore));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuOutputArray;
    cudaCreateSurfaceObject(&outputSurfRef, &resDesc);
}

int main() {
    try {
        
        returnGPUCudaInfoResources(0);
        //const std::string filename = ".\\standard_test_images\\standard_test_images\\lena_color_256.tif";
        const std::string filename = ".\\standard_test_images\\standard_test_images\\lena_color_512.tif";
        //const std::string filename = ".\\standard_test_images\\standard_test_images\\sample_line_box.tif";
        cv::Mat image = imread(filename, IMREAD_COLOR);

        if (image.empty()) {
            printf("Cannot read image file: %s\n", filename.c_str());
            return -1;
        }

        initTexture(image.cols, image.rows, image);
        initSurface(image.cols, image.rows);
       
        // Allocate result of transformation in device memory
        uchar* d_output;
        checkCudaErrors(cudaMalloc((void**)&d_output, image.cols * image.rows * sizeof(uchar) * 3));
        float* d_output_float;
        checkCudaErrors(cudaMalloc((void**)&d_output_float, image.cols * image.rows * sizeof(float) ));
        float* d_x_gradient;
        checkCudaErrors(cudaMalloc((void**)&d_x_gradient, image.cols * image.rows * sizeof(float) ));
        float* d_y_gradient;
        checkCudaErrors(cudaMalloc((void**)&d_y_gradient, image.cols * image.rows * sizeof(float) ));
        float* d_intensity_img;
        checkCudaErrors(cudaMalloc((void**)&d_intensity_img, image.cols * image.rows * sizeof(float)));


        uchar* gpuRef;
        gpuRef = (uchar*)malloc(image.cols * image.rows * sizeof(uchar) * 3);
        float* gpuRefFloat;
        gpuRefFloat = (float*)malloc(image.cols * image.rows * sizeof(float) );


        // Invoke kernel
        dim3 dimBlock(16, 16);
        dim3 dimGrid((image.cols + dimBlock.x - 1) / dimBlock.x,
            (image.rows + dimBlock.y - 1) / dimBlock.y);
        printf("Kernel Dimension :\n   Block size : %i , %i \n    Grid size : %i , %i\n",
            dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        // texObjLinear = input image in uchar , outputSurfRef = intermediate surface, d_intensity_img = intermediate array
        loadIntensitySurface <<< dimGrid, dimBlock >>> (texObjLinear, outputSurfRef, d_intensity_img, image.cols, image.rows);
           
        cudaDeviceSynchronize();


        extractGradients <<< dimGrid, dimBlock >>> (d_output, d_output_float, d_x_gradient, d_y_gradient, d_intensity_img,
                                                    texObjLinear, outputSurfRef, image.cols, image.rows);

        checkCudaErrors(cudaMemcpy(gpuRef, d_output, image.cols * image.rows * sizeof(uchar) * 3, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(gpuRefFloat, d_output_float, image.cols * image.rows * sizeof(float), cudaMemcpyDeviceToHost));
        

        //cv::Mat imageOut = cv::Mat(image.rows, image.cols, CV_8UC3, gpuRef);
        cv::Mat imageOut = image;

                            
        
        std::vector<std::pair<uint32_t, std::pair<int, int>>> sortedListGradients;
        std::vector<std::pair<float, std::pair<int, int>>> sortedListGradientsFloat;
        for (size_t i = 0; i < image.rows; ++i) {
            for (size_t j = 0; j < image.cols; ++j) {

                sortedListGradients.push_back(std::pair<uint32_t, std::pair<int, int>>(gpuRef[(i * image.cols * 3) + (j * 3)],
                    std::pair<int, int>(i, j)));
                sortedListGradientsFloat.push_back(std::pair<float, std::pair<int, int>>(gpuRefFloat[(i * image.cols ) + (j )],
                    std::pair<int, int>(i, j)));
            }
        }
        std::sort(sortedListGradients.begin(), sortedListGradients.end(), [](std::pair<uint32_t, std::pair<int, int>>& a,
            std::pair<uint32_t, std::pair<int, int>>& b) { return a.first > b.first; });
        std::sort(sortedListGradientsFloat.begin(), sortedListGradientsFloat.end(), [](std::pair<float, std::pair<int, int>>& a,
            std::pair<float, std::pair<int, int>>& b) { return a.first > b.first; });

        for (int i = 0; i < 1000; ++i) {
            Point pt;
            //pt.y = sortedListGradients[i].second.first ;
            //pt.x = sortedListGradients[i].second.second;
            //std::cout << sortedListGradients[i].first << std::endl;
            pt.y = sortedListGradientsFloat[i].second.first ;
            pt.x = sortedListGradientsFloat[i].second.second;
            //std::cout << sortedListGradientsFloat[i].first << std::endl;

            cv::circle(imageOut, pt, 2, cv::Scalar(255, 255, 255));
        }
        
        CHECK_CV(imwrite(".\\standard_test_images\\standard_test_images\\lena_color_256.png", imageOut));

        if (CV_SUCCESS == true) printf("\nSuccess !\n");

    }
    catch (Exception ex) {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}