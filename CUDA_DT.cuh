#pragma once

#include "Vector2D.cuh"

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

cudaSurfaceObject_t fastSurface;
cudaSurfaceObject_t voroVerticesSurface;
cudaArray* cuOutputArray;
cudaArray* cuOutputArrayVoroV;

__device__ float dist(int2 a, int2 b) {
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

__global__ void copySurfaceToBuffer(cudaSurfaceObject_t surface, unsigned char* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
    
        unsigned int value;
        surf2Dread(&value, surface, x * sizeof(unsigned int), y);

        unsigned char r = (value >> 16) & 0xFF; 
        unsigned char g = (value >> 8) & 0xFF; 
        unsigned char b = value & 0xFF;         
       
        int index = 3 * (y * width + x);

       
        buffer[index] = r;     
        buffer[index + 1] = g; 
        buffer[index + 2] = b; 
    }
}

cv::Mat createImageFromCudaSurface(cudaSurfaceObject_t surface, int width, int height) {
    unsigned char* deviceBuffer;
    size_t bufferSize = width * height * 3 * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc(&deviceBuffer, bufferSize));

   
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    copySurfaceToBuffer << <dimGrid, dimBlock >> > (surface, deviceBuffer, width, height);
    cudaDeviceSynchronize();

    std::vector<unsigned char> hostBuffer(width * height * 3); // 3 for RGB
    checkCudaErrors(cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceBuffer));

    // Create OpenCV image from host buffer
    cv::Mat image(height, width, CV_8UC3, hostBuffer.data());
    return image.clone(); // Clone is necessary if hostBuffer goes out of scope
}

__device__ void kernel_performVoro(cudaSurfaceObject_t  fastSurface, int2 orig, int2 displace, int2* seeds) {
    unsigned int data;
    unsigned int oData;

    surf2Dread(&data, fastSurface, displace.x * sizeof(unsigned int), displace.y, cudaBoundaryModeZero);
    if (data == 0) return;

    surf2Dread(&oData, fastSurface, orig.x * sizeof(unsigned int), orig.y, cudaBoundaryModeZero);
    if (oData == 0) 
        surf2Dwrite(data, fastSurface, orig.x * sizeof(unsigned int), orig.y, cudaBoundaryModeZero);
    else {
        if (dist(orig, seeds[oData-1]) > dist(orig, seeds[data-1])) {
            surf2Dwrite(data , fastSurface, orig.x * sizeof(unsigned int), orig.y, cudaBoundaryModeZero);
        }
    }
}

__global__ void kernel_AssignSeeds(cudaSurfaceObject_t  fastSurface, int2* seeds, size_t numSeeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSeeds) {
        
        int2 seed = seeds[idx];
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int), seed.y, cudaBoundaryModeZero);
        //1+JFA pre fill step
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int) + 1 , seed.y - 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int)     , seed.y - 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int) - 1 , seed.y - 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int) + 1, seed.y, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int) - 1, seed.y, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int) + 1, seed.y + 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int)    , seed.y + 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int) - 1, seed.y + 1, cudaBoundaryModeZero);
    }
}

__global__ void kernel_onePlusJFA(cudaSurfaceObject_t fastSurface, int2* seeds, size_t numSeeds, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int maxDim = max(w, h);
    int numSteps = ceilf(log2f((float)maxDim));

    for (int i = 0; i < numSteps; ++i) {
        int strideX = max(1, w >> (i + 1));
        int strideY = max(1, h >> (i + 1));
        int2 orig = make_int2(x, y);

        int2 displaced = make_int2(orig.x, orig.y - strideY);
        kernel_performVoro(fastSurface, orig, displaced, seeds);
        displaced.x = orig.x + strideX; displaced.y = orig.y ;
        kernel_performVoro(fastSurface, orig, displaced, seeds);
        displaced.x = orig.x; displaced.y = orig.y + strideY;
        kernel_performVoro(fastSurface, orig, displaced, seeds);
        displaced.x = orig.x - strideX; displaced.y = orig.y;
        kernel_performVoro(fastSurface, orig, displaced, seeds);
        __syncthreads();
    }
}

__global__ void kernel_findAndFixIslands(cudaSurfaceObject_t  fastSurface, int2* seeds, size_t numSeeds, int w, int h) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    unsigned int data; unsigned int dataA[8]; unsigned int seedBuckets[500];

    surf2Dread(&data, fastSurface,x * sizeof(unsigned int), y,            cudaBoundaryModeZero);

    surf2Dread(&dataA[0], fastSurface, x * sizeof(unsigned int) - 1 , y - 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[1], fastSurface, x * sizeof(unsigned int)     , y - 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[2], fastSurface, x * sizeof(unsigned int) + 1 , y - 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[3], fastSurface, x * sizeof(unsigned int) - 1 , y,     cudaBoundaryModeZero);
    surf2Dread(&dataA[4], fastSurface, x * sizeof(unsigned int) + 1 , y,     cudaBoundaryModeZero);
    surf2Dread(&dataA[5], fastSurface, x * sizeof(unsigned int) - 1 , y + 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[6], fastSurface, x * sizeof(unsigned int)     , y + 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[7], fastSurface, x * sizeof(unsigned int) + 1 , y + 1, cudaBoundaryModeZero);

    int areaMatches = 0;
    for (size_t i = 0; i < 8; ++i) 
        if (data == dataA[i]) areaMatches++;

    //found island
    if (areaMatches == 0) {

        int mode = -1;
        for (int i = 0; i < 8; ++i) {
            if (dataA[i] != 0) seedBuckets[dataA[i] - 1] ++;
            if (seedBuckets[dataA[i] - 1] > 1) {
                if (mode == -1) mode = dataA[i];
                else if (seedBuckets[dataA[i] - 1] > seedBuckets[mode - 1]) mode = dataA[i];
            }
        }

        if (mode != -1) surf2Dwrite(mode, fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
        else surf2Dwrite(dataA[1], fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    }
}

__global__ void kernel_findVoroVertices(cudaSurfaceObject_t  fastSurface, int2* seeds, size_t numSeeds, int w, int h,
                                        cudaSurfaceObject_t voroVerticesSurface) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w -1 || y >= h -1) return;

    int data1, data2, data3, data4;

    surf2Dread(&data1, fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    surf2Dread(&data2, fastSurface, x * sizeof(unsigned int) + 1, y, cudaBoundaryModeZero);
    surf2Dread(&data3, fastSurface, x * sizeof(unsigned int), y +1, cudaBoundaryModeZero);
    surf2Dread(&data4, fastSurface, x * sizeof(unsigned int)+1, y+1, cudaBoundaryModeZero);

    //not a voro vertex
    if (data1 == data4 || data2 == data3) return;

    //a voro vertex
    if (data1 != data2 && data2 != data4) {
        int2 voroVertex = make_int2(x, y);

        
    }
    //voro vertex
    else if (data1 != data3 && data3 != data4) {
        int2 voroVertex = make_int2(x, y);
        
    }
}


void initSurface(size_t w, size_t h) {
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&cuOutputArray, &channelDesc, w, h, cudaArraySurfaceLoadStore));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuOutputArray;
    checkCudaErrors(cudaCreateSurfaceObject(&fastSurface, &resDesc));

    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&cuOutputArrayVoroV, &channelDesc, w, h, cudaArraySurfaceLoadStore));

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuOutputArrayVoroV;
    checkCudaErrors(cudaCreateSurfaceObject(&voroVerticesSurface, &resDesc));
}

void CUDA_DT(std::vector<std::pair<int, int>>& hashPoints, cv::Mat& image, cv::Mat& outImage) {

	size_t numSeeds = hashPoints.size();
    int w = image.cols; int h = image.rows;

	if (numSeeds == 0) return;

    initSurface(w, h);

    int2* hostSeeds = (int2*)malloc(numSeeds * sizeof(int2));
    for (size_t i = 0; i < numSeeds; ++i) {
        hostSeeds[i].x = hashPoints[i].second;
        hostSeeds[i].y = hashPoints[i].first;
    }
    int2* deviceSeeds; int2* voroVertices; int* numVoroV;
    
    checkCudaErrors(cudaMalloc((void**)&deviceSeeds, numSeeds * sizeof(int2)));
    checkCudaErrors(cudaMalloc((void**)&voroVertices, 1000 * sizeof(int2)));
    checkCudaErrors(cudaMalloc((void**)&numVoroV, 1 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(deviceSeeds, hostSeeds, numSeeds * sizeof(int2), cudaMemcpyHostToDevice));
    free(hostSeeds);

    {
        dim3 dimBlock(512);
        dim3 dimGrid((numSeeds + dimBlock.x - 1) / dimBlock.x);

        kernel_AssignSeeds << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds);
        checkCudaErrors(cudaDeviceSynchronize());

        cv::Mat imageOut = createImageFromCudaSurface(fastSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT1.png", imageOut);
    }

    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
         
        kernel_onePlusJFA << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds, w, h);
        checkCudaErrors(cudaDeviceSynchronize());

        cv::Mat imageOut = createImageFromCudaSurface(fastSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT2.png", imageOut);
    }

    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);

        kernel_findAndFixIslands << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds, w, h);
        cudaDeviceSynchronize();
    }

    {
        dim3 dimBlock(512, 512);
        dim3 dimGrid(1);

        kernel_findVoroVertices << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds, w, h, voroVerticesSurface);
        cudaDeviceSynchronize();
    }


    cudaDestroySurfaceObject(fastSurface);
    cudaDestroySurfaceObject(voroVerticesSurface);
    cudaFreeArray(cuOutputArray);
    cudaFreeArray(cuOutputArrayVoroV);
    cudaFree(voroVertices);
    cudaFree(numVoroV);
    cudaFree(deviceSeeds);
}