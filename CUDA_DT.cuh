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

cudaArray* cuOutputArray;
cudaArray* cuOutputArrayAlt;
cudaArray* cuOutputArrayVoroV;

__device__ double dist(const int2 a, const int2 b) {
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrt((double)(dx * dx + dy * dy));
}

inline double pointLineSideTest(const int2& A, const int2& B, const int2& P) {
    return (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x);
}

__device__ bool doSegmentsIntersect(const Vec2_CU& A, const Vec2_CU& B, const Vec2_CU& C, const Vec2_CU& D) {
    // Compute direction vectors
    Vec2_CU BA = { B.x - A.x, B.y - A.y };
    Vec2_CU DC = { D.x - C.x, D.y - C.y };
    Vec2_CU CA = { C.x - A.x, C.y - A.y };

    // Compute determinants
    float det = BA.x * DC.y - BA.y * DC.x;
    if (det == 0.0) {
        // Lines are parallel or coincident
        return false;
    }

    // Compute parameters t and u
    float t = (CA.x * DC.y - CA.y * DC.x) / det;
    float u = (CA.x * BA.y - CA.y * BA.x) / det;

    // Check if t and u lie within the range [0, 1]
    return (t >= 0 && t <= 1 && u >= 0 && u <= 1);
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

__device__ void kernel_performVoro(cudaSurfaceObject_t readSurface, cudaSurfaceObject_t writeSurface, int2 orig, int stride, int2* seeds) {
    unsigned int displacedData;
    unsigned int oData;

    int x_moves[4] = { 0, 1, 0 , -1 }; int y_moves[4] = { -1, 0, 1, 0 };
    surf2Dread(&oData, readSurface, orig.x * sizeof(unsigned int), orig.y, cudaBoundaryModeZero);
    unsigned int finalVal = oData;

    for (size_t i = 0; i < 4; ++i) {
        int new_x = orig.x + (stride * x_moves[i]); int new_y = orig.y + (stride * y_moves[i]);
        surf2Dread(&displacedData, readSurface, new_x * sizeof(unsigned int), new_y, cudaBoundaryModeZero);
        if (displacedData == 0) continue;
        if (finalVal == 0) finalVal = displacedData;
        else {
            int seedNum = displacedData - 1; int seedNumOrig = finalVal - 1;
            if (dist(orig, seeds[seedNumOrig]) > dist(orig, seeds[seedNum])) 
                finalVal = displacedData;
        }
    }
    surf2Dwrite(finalVal, writeSurface, orig.x * sizeof(unsigned int), orig.y, cudaBoundaryModeZero);
}

__global__ void kernel_AssignSeeds(cudaSurfaceObject_t  fastSurface, int2* seeds, size_t numSeeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSeeds) {
        
        int2 seed = seeds[idx];
        surf2Dwrite(idx + 1, fastSurface, seed.x * sizeof(unsigned int), seed.y, cudaBoundaryModeZero);

        //1+JFA pre fill step
        surf2Dwrite(idx + 1, fastSurface, (seed.x+1) * sizeof(unsigned int) , seed.y - 1,   cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x)   * sizeof(unsigned int) , seed.y - 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x-1) * sizeof(unsigned int) , seed.y - 1,   cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x+1) * sizeof(unsigned int) , seed.y,       cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x-1) * sizeof(unsigned int) , seed.y,       cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x+1) * sizeof(unsigned int) , seed.y + 1,   cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x)   * sizeof(unsigned int) , seed.y + 1, cudaBoundaryModeZero);
        surf2Dwrite(idx + 1, fastSurface, (seed.x-1) * sizeof(unsigned int) , seed.y + 1,   cudaBoundaryModeZero);
    }
}

__global__ void kernel_onePlusJFA(cudaSurfaceObject_t fastSurfaceRead, cudaSurfaceObject_t fastSurfaceWrite, int2* seeds, size_t numSeeds,
                                  int w, int h, int stride) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int2 orig = make_int2(x, y);
    kernel_performVoro(fastSurfaceRead, fastSurfaceWrite, orig, stride, seeds);
}

__global__ void kernel_findAndFixIslands(cudaSurfaceObject_t fastSurface, int2* seeds, size_t numSeeds, int w, int h) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    unsigned int data; unsigned int dataA[8]; unsigned int seedBuckets[500];

    surf2Dread(&data, fastSurface,x * sizeof(unsigned int), y,            cudaBoundaryModeZero);

    surf2Dread(&dataA[0], fastSurface, (x-1) * sizeof(unsigned int)  , y - 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[1], fastSurface, x * sizeof(unsigned int)     , y - 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[2], fastSurface, (x+1) * sizeof(unsigned int) , y - 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[3], fastSurface, (x-1) * sizeof(unsigned int) , y,     cudaBoundaryModeZero);
    surf2Dread(&dataA[4], fastSurface, (x+1) * sizeof(unsigned int)  , y,     cudaBoundaryModeZero);
    surf2Dread(&dataA[5], fastSurface, (x-1) * sizeof(unsigned int)  , y + 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[6], fastSurface, x * sizeof(unsigned int)     , y + 1, cudaBoundaryModeZero);
    surf2Dread(&dataA[7], fastSurface, (x+1) * sizeof(unsigned int)  , y + 1, cudaBoundaryModeZero);

    int areaMatches = 0;
    for (size_t i = 0; i < 8; ++i) {
        if (data == dataA[i]) areaMatches++;
    }
    //found island
    if (areaMatches == 0) {
        //printf("Island found %i , %i", x, y);
        
        int mode = -1;
        for (int i = 0; i < 8; ++i) {
            if (dataA[i] != 0)   seedBuckets[dataA[i] - 1] ++;
            if (dataA[i] != 0 && seedBuckets[dataA[i] - 1] > 1) {
                if (mode == -1) mode = dataA[i];
                else if (seedBuckets[dataA[i] - 1] > seedBuckets[mode - 1]) mode = dataA[i];
            }
        }

        if (mode != -1) surf2Dwrite(mode, fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
        else surf2Dwrite(dataA[1], fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    }
}

__global__ void kernel_findVoroVertices(cudaSurfaceObject_t  fastSurface, int2* seeds, size_t numSeeds, int w, int h,
                                        cudaSurfaceObject_t voroVerticesSurface, int * getNumTriangles) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int red = 0xFF0000FF;
    unsigned int blue = 0x000000FF;

    if (x >= w -1 || y >= h -1) return;

    unsigned int data1; unsigned int data2; unsigned int data3; unsigned int data4;

    surf2Dread(&data1, fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    surf2Dread(&data2, fastSurface, (x+1) * sizeof(unsigned int), y, cudaBoundaryModeZero);
    surf2Dread(&data3, fastSurface, x * sizeof(unsigned int), y +1, cudaBoundaryModeZero);
    surf2Dread(&data4, fastSurface, (x+1) * sizeof(unsigned int) , y + 1, cudaBoundaryModeZero);

    //not a voro vertex
    if (data1 == data4 || data2 == data3) return;

    //a voro vertex
    if (data1 != data2 && data1 != data3 && data2 != data4 && data3 != data4) {
        atomicAdd(getNumTriangles, 2);
        surf2Dwrite(*getNumTriangles, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    }
    else if (data1 != data2 && data2 != data4) {
        atomicAdd(getNumTriangles, 1);
        surf2Dwrite(*getNumTriangles, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);      
    }
    else if (data1 != data3 && data3 != data4) {
        atomicAdd(getNumTriangles, 1);
        surf2Dwrite(*getNumTriangles, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    }
}

__global__ void kernel_findVoroVerticesCH(cudaSurfaceObject_t voroVerticesSurface, int2* seeds, size_t numSeeds, int w, int h,
     int* getNumTriangles, int2 * CHDevice, int numCH ) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int red = 0xFF0000FF;

    if (x >= w  || y >= h ) return;

    if (x == 305 && y == 210) {
        int fgdgf = 3;
    }

    unsigned int vSiteData; Vec2_CU farPoint1(-1., -1.);
    surf2Dread(&vSiteData, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);

    if (vSiteData == 0) return;
    int2 thisPoint = { x, y }; Vec2_CU thisPointVec2(x, y);

    int countIntersections = 0;

    for (int i = 0; i < numCH - 1; i++) {
        Vec2_CU startP(CHDevice[i].x, -CHDevice[i].y);
        Vec2_CU endP(CHDevice[i + 1].x, -CHDevice[i + 1].y);

        if (doSegmentsIntersect(thisPointVec2, farPoint1, startP, endP)) countIntersections++;
    }

    if (countIntersections == 1) {
        atomicAdd(getNumTriangles, 1);
        surf2Dwrite(red, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    }
    else surf2Dwrite(0, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
}

void initSurface(size_t w, size_t h, cudaSurfaceObject_t * _inSurface, cudaArray * _cuOutputArray) {
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&_cuOutputArray, &channelDesc, w, h, cudaArraySurfaceLoadStore));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = _cuOutputArray;
    checkCudaErrors(cudaCreateSurfaceObject(_inSurface, &resDesc));
}

void CUDA_DT(std::vector<std::pair<int, int>>& hashPoints, cv::Mat& image, cv::Mat& outImage) {

	size_t numSeeds = hashPoints.size();
    int w = image.cols; int h = image.rows;

	if (numSeeds == 0) return;

    cudaSurfaceObject_t fastSurface;
    cudaSurfaceObject_t fastSurfaceAlt;
    cudaSurfaceObject_t voroVerticesSurface;
    
    //set up ping-pong buffer
    initSurface(w, h, &fastSurface, cuOutputArray);
    initSurface(w, h, &fastSurfaceAlt, cuOutputArrayAlt);
    initSurface(w, h, &voroVerticesSurface, cuOutputArrayVoroV);

    int2* hostSeeds = (int2*)malloc(numSeeds * sizeof(int2));
    for (size_t i = 0; i < numSeeds; ++i) {
        hostSeeds[i].x = hashPoints[i].second;
        hostSeeds[i].y = hashPoints[i].first;
    }
    int2* deviceSeeds; int2* voroVertices; int* getNumTriangles; int numVoroTemp = 0;
    std::vector<int2> CH; //points of the convex hull
    int2* CHDevice;
    
    checkCudaErrors(cudaMalloc((void**)&deviceSeeds, numSeeds * sizeof(int2)));
    checkCudaErrors(cudaMalloc((void**)&voroVertices, 1000 * sizeof(int2)));
    checkCudaErrors(cudaMalloc((void**)&getNumTriangles, sizeof(int)));
    checkCudaErrors(cudaMemcpy(deviceSeeds, hostSeeds, numSeeds * sizeof(int2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(getNumTriangles, &numVoroTemp,  sizeof(int), cudaMemcpyHostToDevice));
    
    
    {
        dim3 dimBlock(512);
        dim3 dimGrid((numSeeds + dimBlock.x - 1) / dimBlock.x);

        kernel_AssignSeeds << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds);
        //checkCudaErrors(cudaDeviceSynchronize());

        kernel_AssignSeeds << < dimGrid, dimBlock >> > (fastSurfaceAlt, deviceSeeds, numSeeds);
        checkCudaErrors(cudaDeviceSynchronize());

        cv::Mat imageOut = createImageFromCudaSurface(fastSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT1.png", imageOut);

        cv::Mat imageOut1 = createImageFromCudaSurface(fastSurfaceAlt, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT1alt.png", imageOut);
    }

    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
        //dim3 dimGrid(1,1);
        
        int maxDim = max(w, h);
        int numSteps = ceil(log2(maxDim));

        for (int i = 0; i < numSteps; ++i) {
            int stride = max(1, w >> (i + 1));
            //std::cout << stride << std::endl;

            if (i & 1) {
                kernel_onePlusJFA << < dimGrid, dimBlock >> > (fastSurface, fastSurfaceAlt, deviceSeeds, numSeeds, w, h, stride);
                checkCudaErrors(cudaDeviceSynchronize());
            }
            else {
                kernel_onePlusJFA << < dimGrid, dimBlock >> > (fastSurfaceAlt, fastSurface, deviceSeeds, numSeeds, w, h, stride);
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }

        cv::Mat imageOut = createImageFromCudaSurface(fastSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT2.png", imageOut);
        cv::Mat imageOut1 = createImageFromCudaSurface(fastSurfaceAlt, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT2alt.png", imageOut);
    }
    
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
        
        for (size_t i = 0; i < 5; ++i) {
            kernel_findAndFixIslands << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds, w, h);
            checkCudaErrors(cudaDeviceSynchronize());
        }

        cv::Mat imageOut = createImageFromCudaSurface(fastSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT3.png", imageOut);
    }
     
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);

        kernel_findVoroVertices << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds, w, h, voroVerticesSurface, getNumTriangles);
        checkCudaErrors(cudaDeviceSynchronize());

        int* getNumTrianglesTemp = new int();
        checkCudaErrors(cudaMemcpy(getNumTrianglesTemp, getNumTriangles, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << *getNumTrianglesTemp << std::endl;

        delete ( getNumTrianglesTemp );

        cv::Mat imageOut = createImageFromCudaSurface(voroVerticesSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT4.png", imageOut);
    }
    
    {
        //perform convex hull on original points
        //graham scan, sort w.r.t. x then y axis
        //x and y coords reversed in points
        std::sort(begin(hashPoints), end(hashPoints), [](std::pair<int, int>& a,std::pair<int, int>& b) { 
            return a.second < b.second || (a.second == b.second && a.first < b.first); });
        
        //top half of convex hull
        for (size_t i = 0; i < hashPoints.size(); ++i) {
            int2 P = { hashPoints[i].second, -hashPoints[i].first };
            while (CH.size() >= 2 && pointLineSideTest(CH[CH.size() - 2], CH.back(), P) > 0.0) {
                CH.pop_back();
            }
            CH.push_back(P);
        }

        //bottom half of convex hull
        int bottomBaseSize = CH.size();
        for (int i = hashPoints.size() - 2; i >= 0; --i) {
            int2 P = { hashPoints[i].second, -hashPoints[i].first };
            while (CH.size() >= bottomBaseSize + 1 && pointLineSideTest(CH[CH.size() - 2], CH.back(), P) > 0.0) {
                CH.pop_back();
            }
            CH.push_back(P);
        }

        cv::Mat imageOutCH = image;
        imageOutCH.setTo(cv::Scalar(0, 0, 0));

        for (size_t i = 0; i < CH.size(); ++i) {
            cv::Point pt;
            pt.x = CH[i].x;
            pt.y = -CH[i].y;
            cv::circle(imageOutCH, pt, 2, cv::Scalar(255, 255, 255));
        }

        imwrite(".\\standard_test_images\\standard_test_images\\outputCH.png", imageOutCH);
    }
    {
        //elimate all voronai vertices that are not in the CH
        //those triangles formed from voronoi vertices not in the CH will be added by the CH
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);

        CHDevice = (int2*)malloc(CH.size() * sizeof(int2));
        checkCudaErrors(cudaMalloc((void**)&CHDevice, CH.size() * sizeof(int2)));
        checkCudaErrors(cudaMemcpy(CHDevice, &CH, CH.size() * sizeof(int2), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(getNumTriangles, &numVoroTemp, sizeof(int), cudaMemcpyHostToDevice));

        kernel_findVoroVerticesCH << < dimGrid, dimBlock >> > (voroVerticesSurface, deviceSeeds, numSeeds, w, h, getNumTriangles, CHDevice, CH.size());
        checkCudaErrors(cudaDeviceSynchronize());

        int* getNumTrianglesTemp = new int();
        checkCudaErrors(cudaMemcpy(getNumTrianglesTemp, getNumTriangles, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << *getNumTrianglesTemp << std::endl;

        delete (getNumTrianglesTemp);

        cv::Mat imageOut = createImageFromCudaSurface(voroVerticesSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT5.png", imageOut);
    }

    cudaDestroySurfaceObject(fastSurface);
    cudaDestroySurfaceObject(fastSurfaceAlt);
    cudaDestroySurfaceObject(voroVerticesSurface);
    cudaFreeArray(cuOutputArray);
    cudaFreeArray(cuOutputArrayAlt);
    cudaFreeArray(cuOutputArrayVoroV);
    cudaFree(voroVertices);
    cudaFree(getNumTriangles);
    cudaFree(deviceSeeds);
    free(hostSeeds);
}