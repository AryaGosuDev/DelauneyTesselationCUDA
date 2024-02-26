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

__device__ int reserveIndex(int* globalCounter, int accum) {
    return atomicAdd(globalCounter, accum);
}

__device__ double dist(const int2 a, const int2 b) {
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return sqrt((double)(dx * dx + dy * dy));
}

inline double pointLineSideTest(const int2& A, const int2& B, const int2& P) {
    return (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x);
}

inline size_t returnVoronoiRegion(Vec2 pt, cv::Mat & imageVoronoi) {
    if (imageVoronoi.type() != CV_8UC3) {
        std::cerr << "Incorrect image format." << std::endl;
        return 0;
    }
    pt.y *= -1.0;
    cv::Vec3b& pixelValue = imageVoronoi.at<cv::Vec3b>(pt.y, pt.x);
    size_t blue = pixelValue[0];
    size_t green = pixelValue[1];
    size_t red = pixelValue[2];
    size_t voronoiRegion = (blue << 16) | (green << 8) | red;
    return voronoiRegion;
}

// checks to see if line a->b points are on opposite sides of line c->d, and vice versa
__device__ bool doSegmentsIntersectSideSegment(Vec2_CU &a, Vec2_CU & b, Vec2_CU& c, Vec2_CU &d) {
    double x0 = b.x - a.x;
    double x1 = d.x - c.x;
    double y0 = b.y - a.y;
    double y1 = d.y - c.y;

    double p0 = y1 * (d.x - a.x) - x1 * (d.y - a.y);
    double p1 = y1 * (d.x - b.x) - x1 * (d.y - b.y);
    double p2 = y0 * (b.x - c.x) - x0 * (b.y - c.y);
    double p3 = y0 * (b.x - d.x) - x0 * (b.y - d.y);

    return (p0 * p1 <= 0) & (p2 * p3 <= 0);
}

__device__ bool doSegmentsIntersect(Vec2_CU a, Vec2_CU b, Vec2_CU c, Vec2_CU d) {

    double det  = (b.x - a.x) * (d.y - c.y) - (d.x - c.x) * (b.y - a.y);
    double det1 = (b.x - a.x) * (a.y - c.y) - (a.x - c.x) * (b.y - a.y);
    double det2 = (d.x - c.x) * (a.y - c.y) - (a.x - c.x) * (d.y - c.y);

    if (det == 0.0) return false;

    if ((det1 / det) >= 0.0 && (det1 / det) < 1.0 && (det2 / det) >= 0.0 && (det2 / det) <= 1.0) return true;
    return false;
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
        int uniqueIdent = reserveIndex(getNumTriangles, 2);
        surf2Dwrite(uniqueIdent, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
        return;
    }
    else if (data1 != data2 && data2 != data4) {
        int uniqueIdent = reserveIndex(getNumTriangles, 1);
        surf2Dwrite(uniqueIdent, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
        return;
    }
    else if (data1 != data3 && data3 != data4) {
        int uniqueIdent = reserveIndex(getNumTriangles, 1); 
        surf2Dwrite(uniqueIdent, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
        return;
    }
    surf2Dwrite(0, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    
}

__global__ void kernel_findVoroVerticesCH(cudaSurfaceObject_t voroVerticesSurface, int2* seeds, size_t numSeeds, int w, int h,
     int* getNumTriangles, int2 * CHDevice, int numCH ) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w  || y >= h ) return;

    unsigned int vSiteData; Vec2_CU farPoint1(-1., -1.);
    surf2Dread(&vSiteData, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);

    if (vSiteData == 0) return;
    Vec2_CU thisPointVec2(x, -y);

    int countIntersections = 0;

    for (int i = 0; i < numCH - 1; i++) {
        Vec2_CU startP(CHDevice[i].x, CHDevice[i].y);
        Vec2_CU endP(CHDevice[i + 1].x, CHDevice[i + 1].y);

        if (doSegmentsIntersect(thisPointVec2, farPoint1, startP, endP)) 
            countIntersections++;
    }

    if (countIntersections == 1) {
        //atomicAdd(getNumTriangles, 1);
        surf2Dwrite(vSiteData, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    }
    else surf2Dwrite(0, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
}

__global__ void kernel_CreateDTPrime(cudaSurfaceObject_t fastSurface, int2* seeds, size_t numSeeds, int w, int h,
    cudaSurfaceObject_t voroVerticesSurface, int* getNumTriangles, Triangle_CU * DTPrime, uint32_t* trHash) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w  || y >= h ) return;

    unsigned int data1; unsigned int data2; unsigned int data3; unsigned int data4;
    unsigned int vSiteData;

    surf2Dread(&vSiteData, voroVerticesSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);

    if (vSiteData == 0) return;

    surf2Dread(&data1, fastSurface, x * sizeof(unsigned int), y, cudaBoundaryModeZero);
    surf2Dread(&data2, fastSurface, (x + 1) * sizeof(unsigned int), y, cudaBoundaryModeZero);
    surf2Dread(&data3, fastSurface, x * sizeof(unsigned int), y + 1, cudaBoundaryModeZero);
    surf2Dread(&data4, fastSurface, (x + 1) * sizeof(unsigned int), y + 1, cudaBoundaryModeZero);

    //a voro vertex
    if (data1 != data2 && data1 != data3 && data2 != data4 && data3 != data4) {
        //create 2 triangles
        DTPrime[vSiteData - 1] = Triangle_CU(data1, data3, data4);
        trHash[vSiteData - 1] = DTPrime[vSiteData - 1].commutativeHash;
        DTPrime[vSiteData - 2] = Triangle_CU(data1, data2, data4);
        trHash[vSiteData - 2] = DTPrime[vSiteData - 2].commutativeHash;
    }
    else if (data1 != data2 && data2 != data4) {
        //create 1 tri
        DTPrime[vSiteData - 1] = Triangle_CU(data1, data2, data4);
        trHash[vSiteData - 1] = DTPrime[vSiteData - 1].commutativeHash;
    }
    else if (data1 != data3 && data3 != data4) {
        //create 1 tri
        DTPrime[vSiteData - 1] = Triangle_CU(data1, data3, data4);
        trHash[vSiteData - 1] = DTPrime[vSiteData - 1].commutativeHash;
    }
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
    int2* CHDevice; Triangle_CU* DTPrime; uint32_t  *trHash;
    std::vector< Triangle> DT; std::unordered_map<uint32_t, size_t> triHashMap;
    
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
         //draw CH on image
        cv::Mat imageOutCH = image;
        imageOutCH.setTo(cv::Scalar(0, 0, 0));

        for (size_t i = 0; i < CH.size(); ++i) {
            cv::Point pt;
            pt.x = CH[i].x;
            pt.y = -CH[i].y;
            cv::Point pt2; 
            pt2.x = CH[(i + 1) % CH.size()].x;
            pt2.y = -CH[(i + 1) % CH.size()].y;

            cv::circle(imageOutCH, pt, 2, cv::Scalar(255, 255, 255));
            cv::Scalar lineColor(0, 255, 0); // Green color in BGR format
            int lineWidth = 2; // Line thickness
            cv::line(imageOutCH, pt, pt2, lineColor, lineWidth);
        }

        imwrite(".\\standard_test_images\\standard_test_images\\outputCH.png", imageOutCH);
    }
    {
        //elimate all voronai vertices that are not in the CH
        //those triangles formed from voronoi vertices not in the CH will be added by the CH
        // becuae their voronoi sites are outside of the convex hulls, thus their triangles are formed as part of the CH
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);

        int2* CHHost = (int2*)malloc(CH.size() * sizeof(int2));
        for (int i = 0; i < CH.size(); ++i) 
            CHHost[i] = CH[i];
        
        checkCudaErrors(cudaMalloc((void**)&CHDevice, CH.size() * sizeof(int2)));
        checkCudaErrors(cudaMemcpy(CHDevice, CHHost, CH.size() * sizeof(int2), cudaMemcpyHostToDevice));
        free(CHHost);
        //checkCudaErrors(cudaMemcpy(getNumTriangles, &numVoroTemp, sizeof(int), cudaMemcpyHostToDevice));

        kernel_findVoroVerticesCH << < dimGrid, dimBlock >> > (voroVerticesSurface, deviceSeeds, numSeeds, w, h, 
                                                               getNumTriangles, CHDevice, CH.size());
        checkCudaErrors(cudaDeviceSynchronize());

        int* getNumTrianglesTemp = new int();
        checkCudaErrors(cudaMemcpy(getNumTrianglesTemp, getNumTriangles, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << *getNumTrianglesTemp << std::endl;
        delete (getNumTrianglesTemp);

        cv::Mat imageOut = createImageFromCudaSurface(voroVerticesSurface, w, h);
        imwrite(".\\standard_test_images\\standard_test_images\\outputDT5.png", imageOut);
    }
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);

        int* getNumTrianglesTemp = new int();

        checkCudaErrors(cudaMemcpy(getNumTrianglesTemp, getNumTriangles, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMalloc((void**)&DTPrime, (*getNumTrianglesTemp) * sizeof(Triangle_CU)));
        checkCudaErrors(cudaMalloc((void**)&trHash, (*getNumTrianglesTemp) * sizeof(uint32_t)));

        kernel_CreateDTPrime << < dimGrid, dimBlock >> > (fastSurface, deviceSeeds, numSeeds, w, h,
            voroVerticesSurface, getNumTriangles, DTPrime, trHash);
        checkCudaErrors(cudaDeviceSynchronize());

        delete (getNumTrianglesTemp);
    }
    {
        //convert the DT` to DT by following the CH points and completing the DT
        int* getNumTrianglesTemp = new int();
        checkCudaErrors(cudaMemcpy(getNumTrianglesTemp, getNumTriangles, sizeof(int), cudaMemcpyDeviceToHost));

        Triangle* DTPrimeHost; uint32_t* trHashHost;
        std::cout << sizeof(Triangle) << " " << sizeof(Triangle_CU);
        DTPrimeHost = (Triangle*)malloc(*getNumTrianglesTemp * sizeof(Triangle));
        trHashHost = (uint32_t*)malloc(*getNumTrianglesTemp * sizeof(uint32_t));
        std::vector< Triangle> DT; std::unordered_map<uint32_t, size_t> triHashMap;

        checkCudaErrors(cudaMemcpy(DTPrimeHost, DTPrime, *getNumTrianglesTemp * sizeof(Triangle), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(trHashHost, trHash, *getNumTrianglesTemp * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < *getNumTrianglesTemp; ++i) {

            DT.push_back(DTPrimeHost[i]); 
            size_t repeatedTri = 0;
            if (triHashMap.find(DT.back().commutativeHash) != end(triHashMap)) {
                if (DT.back().commutativeHash != 0) {
                    repeatedTri = triHashMap[DT.back().commutativeHash];
                }
                std::cout << "repeated hash : " << DT.back().commutativeHash << std::endl;
            }
            else triHashMap[DT.back().commutativeHash] = i;
        }

        free(DTPrimeHost); free(trHashHost);

        //now follow CH around to add missing triangles that were not added as
        //vertices in the Voronoi sites
        const std::string filename = ".\\standard_test_images\\standard_test_images\\outputDT2.png";
        cv::Mat imageVoronoi = imread(filename, cv::IMREAD_COLOR);

        for (size_t i = 0; i < CH.size() - 1; ++i) {
            std::vector<size_t> s;
            
            Vec2 pt1(CH[i].x, CH[i].y); Vec2 pt2(CH[i + 1].x, CH[i + 1].y);
            s.push_back(returnVoronoiRegion(pt1, imageVoronoi));

            int dx = abs(pt2.x - pt1.x); double stepX = 0; double stepY = 0;
            if (pt1.x < pt2.x) stepX = 1;
            else stepX = -1;

            int dy = -(abs(pt2.y - pt1.y));
            if (pt1.y < pt2.y) stepY = 1;
            else stepY = -1;

            double e = dx + dy;
            while (1) {
                size_t vRegion = returnVoronoiRegion(pt1, imageVoronoi);
                if (s.size() == 1 && s[0] != vRegion) s.push_back(vRegion);
                else if (s.size() == 2) {  //possible new triangle
                    if (s[1] != vRegion) {
                        s.push_back(vRegion);
                        Triangle newTri(s[0], s[1], s[2]);
                        size_t repeatedTri = 0;
                        // already encountered this tri, remove top of stack then
                        if (triHashMap.find(newTri.commutativeHash) != end(triHashMap)) {
                            if (newTri.commutativeHash != 0) {
                                repeatedTri = triHashMap[newTri.commutativeHash];
                            }
                            std::cout << "repeated hash CH: " << newTri.commutativeHash << std::endl;
                        }
                        else { // new tri detected
                            DT.push_back(newTri);
                            triHashMap[newTri.commutativeHash] = DT.size() - 1;

                        }
                    }
                }
                else if (s.size() == 3) {
                    if (s[2] != vRegion) {
                        s[1] = s[2]; s[2] = vRegion;
                        Triangle newTri(s[0], s[1], s[2]);
                        size_t repeatedTri = 0;
                        // already encountered this tri, remove top of stack then
                        if (triHashMap.find(newTri.commutativeHash) != end(triHashMap)) {
                            if (newTri.commutativeHash != 0) {
                                repeatedTri = triHashMap[newTri.commutativeHash];
                            }
                            std::cout << "repeated hash CH: " << newTri.commutativeHash << std::endl;
                        }
                        else { // new tri detected
                            DT.push_back(newTri);
                            triHashMap[newTri.commutativeHash] = DT.size() - 1;
                        }
                    }
                }
                if (pt1.x == pt2.x && pt1.y == pt2.y) break;
                double e2 = 2.0 * e;
                if (e2 >= dy) {
                    if (pt1.x == pt2.x) break;
                    e += dy; pt1.x += stepX;
                }
                if (e2 <= dx) {
                    if (pt1.y == pt2.y) break;
                    e += dx; pt1.y += stepY;
                }
            }
        }

        cv::Mat imageOutDT = image;
        imageOutDT.setTo(cv::Scalar(0, 0, 0));

        int2* hostSeeds = (int2*)malloc(numSeeds * sizeof(int2));
        checkCudaErrors(cudaMemcpy(hostSeeds, deviceSeeds, numSeeds * sizeof(int2), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < DT.size(); ++i) {
            if (DT[i].commutativeHash != 0) {
                cv::Point pt1;
                pt1.x = hostSeeds[DT[i].v1_indx - 1].x;
                pt1.y = hostSeeds[DT[i].v1_indx - 1].y;
                cv::Point pt2;
                pt2.x = hostSeeds[DT[i].v2_indx - 1].x;
                pt2.y = hostSeeds[DT[i].v2_indx - 1].y;
                cv::Point pt3;
                pt3.x = hostSeeds[DT[i].v3_indx - 1].x;
                pt3.y = hostSeeds[DT[i].v3_indx - 1].y;
                
                cv::Scalar lineColor(0, 255, 0); // Green color in BGR format
                int lineWidth = 1; // Line thickness
                cv::line(imageOutDT, pt1, pt2, lineColor, lineWidth);
                cv::line(imageOutDT, pt2, pt3, lineColor, lineWidth);
                cv::line(imageOutDT, pt3, pt1, lineColor, lineWidth);

            }
        }

        imwrite(".\\standard_test_images\\standard_test_images\\outputDTFinal.png", imageOutDT);

        delete (getNumTrianglesTemp);
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