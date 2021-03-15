#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "setOp.h"
using namespace std;
// #define DEBUG 1

#define checkCudaError(a)                                                        \
    do                                                                           \
    {                                                                            \
        if (cudaSuccess != (a))                                                  \
        {                                                                        \
            fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n",                                                                    \
                    __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

int main(int argc, char **argv)
{
    int numData = atoi(argv[1]); // the number of parallel threads
    int lists = atoi(argv[2]);
    int modW = atoi(argv[3]);            
    int threads = diWarp(numData);
    int totalLen = 0;
    std::vector<int> sortNo, state, permute, cast;
    int *out = new int[1];
    int *devOut = 0;
    int *listNum = new int[lists + 1];
    int *devListNum = 0;
    int *testData = 0;
    // int *oriData;
    int *devSrcData = 0;
    int **dataPointer = new int *[MAXSHARE];
    int **devDataPointer = 0;
    int *castData, *sortData;
    int *devSortData, *devBoarder, *devCastData, *devOriData;
    bool check = true;
    listNum[0] = 0;
    cudaMalloc((void **)&devDataPointer, sizeof(int *) * MAXSHARE);
    cudaMalloc((void **)&devListNum, sizeof(int *) * (lists + 1));
    srand(0);

    for (int i = 0; i < lists; i++)
    {
        int length = rand() % (numData / 2 + 1);
        int *host = new int[length];
        for (int j = 0; j < length; j++)
        {
            host[j] = rand() % (numData / 2 + 1);
            // printf("%d ", host[j]);
            sortNo.push_back(host[j]);
        }
        // printf("\n");
        int *dev;
        cudaMalloc((void **)&dev, sizeof(int) * length);
        cudaMemcpy(dev, host, sizeof(int) * length, cudaMemcpyHostToDevice);
        dataPointer[i] = dev;
        totalLen += length;
        listNum[i + 1] = totalLen;
        delete[] host;
    }
    state.assign(sortNo.begin(), sortNo.end());
    std::sort(state.begin(), state.end());
    vector<int>::iterator it;
    it = std::unique(state.begin(), state.end());
    state.erase(it, state.end());
    for(int i = 0;i < sortNo.size();i++)
    {
        permute.push_back(i % modW);
    }
    for(auto it : sortNo)
    {
        vector<int>::iterator iter;
        iter = std::find(state.begin(), state.end(), it);
        cast.push_back(std::distance(state.begin(),iter) % modW);
    }
    printf("\n");
    printf("totalLen: %d\n", totalLen);
    int* boarder = (int *)malloc(sizeof(int) * CEIL(totalLen, warpSize));
    // oriData = (int *)malloc(sizeof(int) * totalLen);
    testData = (int *)malloc(sizeof(int) * totalLen);
    castData = (int *)malloc(totalLen * sizeof(int));
    sortData = (int *)malloc(totalLen * sizeof(int));
    int blockNum = CEIL(totalLen, warpSize);
    cudaMemcpy(devListNum, listNum, sizeof(int *) * (lists + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(devDataPointer, dataPointer, sizeof(int *) * lists, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devSrcData, sizeof(int *) * totalLen);
    cudaMalloc((void **)&devOut, sizeof(int));
    cudaMalloc((void **)&devSortData, sizeof(int) * totalLen);
    cudaMalloc((void **)&devCastData, sizeof(int) * totalLen);
    cudaMalloc((void **)&devOriData, sizeof(int) * totalLen);
    cudaMalloc((void **)&devBoarder, sizeof(int) * (1 + CEIL(totalLen, warpSize)));
    
    setOp(devDataPointer, devOriData, devSrcData, devSortData, devCastData, devOut,
          devListNum, devBoarder, threads, lists, totalLen, blockNum, modW);

    cudaMemcpy(castData, devCastData, sizeof(int) * totalLen, cudaMemcpyDeviceToHost);
    cudaMemcpy(sortData, devSortData, sizeof(int) * totalLen, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, devOut, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(boarder, devBoarder, sizeof(int) * (1 + CEIL(totalLen, warpSize)), cudaMemcpyDeviceToHost);

    for (int i = 0; i < totalLen; i++)
    {
        if(cast[i] != castData[i])
        {
            printf("%d %d %d \n", cast[i], castData[i], i);
            printf("error\n");
            check = false;
            break;
        }
    }
    printf("\n");
    if(check)
        printf("check!\n");

    for (int i = 0; i < lists; i++)
    {
        int length = listNum[i + 1] - listNum[i];
        int *host = new int[length];
        cudaMemcpy(host, dataPointer[i], sizeof(int) * length, cudaMemcpyDeviceToHost);
        dataPointer[i] = host;
        // for (int j = 0; j < length; j++)
        // {
        //     printf("%d ", dataPointer[i][j]);
        // }
        // printf("\n");
    }
    for (int i = 0; i < lists; i++)
    {
        delete [] dataPointer[i];
        cudaFree(&devDataPointer[i]);
    }
   
    printf("\n");

    delete[] out;
    delete[] dataPointer;
    delete[] listNum;
    free(testData);
    free(sortData);
    free(castData);
    free(boarder);

    cudaFree(devOut);                 
    cudaFree(devSortData);
    cudaFree(devBoarder);
    cudaFree(devCastData);
    cudaFree(devOriData);
    cudaFree(devListNum);
    cudaFree(devDataPointer);
    cudaFree(devSrcData);
}