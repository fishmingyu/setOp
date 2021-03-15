#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define MAX_NUM_LISTS 100
#define MAX_THREADS 64
#define warpSize 32
#define MAXLISTS 100
#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)
#define diWarp(x) (MIN((CEIL(x, 32) << 5), MAX_THREADS))
#define MAXSEC 128

__device__ __forceinline__ int biSearch(const int *srcData, int data, int start, int end);

__global__ void joint(int **const dataPointer, int *destData, int *listNum)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int numInList = listNum[bid + 1] - listNum[bid];
    for (int i = 0; i < CEIL(numInList, blockDim.x); i++)
    {
        if (tid < numInList)
        {
            destData[listNum[bid] + tid] = dataPointer[bid][tid];
        }
        tid += blockDim.x;
    }
}

__global__ void split(int **const dataPointer, int *const srcData, int *listNum)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int numInList = listNum[bid + 1] - listNum[bid];
    for (int i = 0; i < CEIL(numInList, blockDim.x); i++)
    {
        if (tid < numInList)
        {
            dataPointer[bid][tid] = srcData[listNum[bid] + tid];
        }
        tid += blockDim.x;
    }
}

__device__ void radixSort(int *const srcData, int *const destData,
                          int numLists, int numData, int tid)
{
    for (int bit = 0; bit < 32; bit++)
    {
        int mask = (1 << bit);
        int count0 = 0;
        int count1 = 0;
        for (int i = tid; i < numData; i += numLists)
        {
            unsigned int *temp = (unsigned int *)&srcData[i];
            if (*temp & mask)
            {
                destData[tid + count1 * numLists] = srcData[i];
                count1 += 1;
            }
            else
            {
                srcData[tid + count0 * numLists] = srcData[i];
                count0 += 1;
            }
        }
        for (int j = 0; j < count1; j++)
        {
            srcData[tid + count0 * numLists + j * numLists] = destData[tid + j * numLists];
        }
    }
}

__device__ void mergeDedup(const int *srcData, int *const dest_list, int numLists, int numData, int *outNum, int tid)
{
    int nPerList = CEIL(numData, numLists);
    __shared__ int listIdx[MAX_NUM_LISTS];
    __shared__ int reducVal[MAX_NUM_LISTS];
    __shared__ int reducIdx[MAX_NUM_LISTS];

    listIdx[tid] = 0;
    reducVal[tid] = 0;
    reducIdx[tid] = 0;

    __syncthreads();
    int max;
    int count = 0;
    outNum[0] = numData;
    for (int i = 0; i < numData; i++)
    {
        unsigned int tmp_data = INT_MAX;
        if (listIdx[tid] < nPerList)
        {
            int src_index = tid + listIdx[tid] * numLists;
            if (src_index < numData)
            {
                tmp_data = srcData[src_index];
            }
        }
        reducIdx[tid] = tid;
        reducVal[tid] = tmp_data;
        __syncthreads();
        //it may be a good idea to use __reduce_min_sync or _shfl_down_sync to substitute this share memory method,
        //but it will take some time to debug
        for (int tid_max = (numLists >> 1); tid_max > 0; tid_max >>= 1)
        {
            if (tid < tid_max)
            {
                const int tmp_idx = tid + tid_max;
                const int val = reducVal[tmp_idx];
                if (val < reducVal[tid])
                {
                    reducVal[tid] = val;
                    reducIdx[tid] = reducIdx[tmp_idx];
                }
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            if (max == reducVal[0])
            {
                count--;
                outNum[0]--;
            }
            listIdx[reducIdx[0]]++;
            dest_list[count] = reducVal[0];
            count++;
            max = reducVal[0];
        }
        __syncthreads();
    }
}

__global__ void sortAndDedep(int *const srcData, int *const destData,
                             int numLists, int numData, int *outNum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    radixSort(srcData, destData, numLists, numData, tid);
    __syncthreads();
    mergeDedup(srcData, destData, numLists, numData, outNum, tid);
}

__global__ void order(int *const srcData, int *const boarder,
                      int *numData, int modW)
{
    int tid = threadIdx.x;
    int secNum = CEIL(numData[0], warpSize);
    if (tid > 0 && tid < secNum)
    {
        if (warpSize * tid < numData[0])
            boarder[tid] = srcData[warpSize * tid];
    }
    else if (tid == 0)
    {
        boarder[tid] = 0;
        boarder[secNum] = srcData[numData[0] - 1];
    }
}

__global__ void find(int *const srcData, int *const orderData, int *const destData, int *const boarderNum, int total, int *numData, int modW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total)
    {
        int idx = biSearch(boarderNum, srcData[tid], 0, CEIL(numData[0], warpSize));
        int idx2 = biSearch(orderData, srcData[tid], idx * warpSize, MIN(((idx + 1) * warpSize), numData[0] - 1));
        destData[tid] = idx2 % modW;
        int idx3 = biSearch(orderData, srcData[tid], 0, numData[0] - 1);
    }
}

__device__ __forceinline__ int biSearch(const int *srcData, int data, int start, int end)
{
    int low = start, high = end;
    if (low == high)
        return low;
    while (low < high)
    {
        int mid = (low + high) >> 1;
        if (srcData[mid] <= data)
            low = mid + 1;
        else
            high = mid;
    }
    if (srcData[high] == data)
        return high;
    else
        return high - 1;
}

__host__ void setOp(int **const devDataPointer, int *devOriData, int *devSrcData, int *devSortData,
                    int *devCastData, int *devOut, int *devListNum, int *devBoarder,
                    int numLists, int lists, int totalLen, int blockNum, int modW)
{
    joint<<<lists, warpSize>>>(devDataPointer, devSrcData, devListNum);
    cudaMemcpy(devOriData, devSrcData, sizeof(int) * totalLen, cudaMemcpyDeviceToDevice);
    sortAndDedep<<<1, numLists>>>(devSrcData, devSortData, numLists, totalLen, devOut);
    order<<<1, MIN(MAX(blockNum, 2), MAXSEC)>>>(devSortData, devBoarder, devOut, modW);
    find<<<blockNum, warpSize>>>(devOriData, devSortData, devCastData, devBoarder, totalLen, devOut, modW);
    split<<<lists, warpSize>>>(devDataPointer, devCastData, devListNum);
}
