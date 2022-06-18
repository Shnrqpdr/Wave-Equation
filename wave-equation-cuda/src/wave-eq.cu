
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 500 // Tamanho da Malha
#define dom 50.0
#define IT 10 // Loops temporais
#define alpha 1.4
#define gamma 1.8
#define xInicial 0
#define xFinal 50.0
#define yInicial 0
#define yFinal 50.0

#define CHECK(call)                                                \
    {                                                              \
        cudaError_t error = call;                                  \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

__global__ void waveEquationKernel(double *wave, double *waveFuture, double *wavePast)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1))
        waveFuture[i * blockDim.x + j] = 2 * wave[i * blockDim.x + j] * (1 - alpha * alpha - gamma * gamma) - wavePast[i * blockDim.x + j] + alpha * alpha * wave[(i + 1) * blockDim.x + j] + alpha * alpha * wave[(i - 1) * blockDim.x + j] + gamma * gamma * wave[i * blockDim.x + (j + 1)] + gamma * gamma * wave[i * blockDim.x + (j - 1)];
}

__global__ void actualizationKernel(double *wave, double *waveFuture, double *wavePast)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1))
    {
        wavePast[i * blockDim.x + j] = wave[i * blockDim.x + j];
        wave[i * blockDim.x + j] = waveFuture[i * blockDim.x + j];
    }
}

void inicMatrix(double *wave)
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            wave[i * N + j] = 0;
        }
    }
}

void writeFiles(double *wave, double dx, double dy)
{
    int i, j;
    FILE *fileStaticPlot;

    fileStaticPlot = fopen("WaveStatic.dat", "w");

    fprintf(fileStaticPlot, "x\ty\tt\tf\n");

    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            fprintf(fileStaticPlot, "%lf\t%lf\t%lf\n", i * dx, j * dy, wave[i * N + j]);
        }
    }

    fclose(fileStaticPlot);
}

void initialCond(double *wave)
{
    int i, j;
    // double x, y;
    // double dx, dy;

    // y = -dom / 2.0;
    for (j = 0; j < N; j++)
    {
        // x = -dom / 2.0;
        for (i = 0; i < N; i++)
        {
            wave[i * N + j] = 4 * sin(M_PI * i / 25.0);

            // wave[i * N + j] = 2;
        }
    }
}

void deviceCapabilities()
{

    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf(" --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execition timeout : ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf(" --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf(" --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n",
               prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("\n");
    }
}

int main()
{
    double *hostWave, *hostWaveFuture, *hostWavePast;       // Host variables
    double *deviceWave, *deviceWaveFuture, *deviceWavePast; // Device Variables
    int i, j, k;
    double dx, dy;

    dx = (xFinal - xInicial) / N;
    dy = (yFinal - yInicial) / N;

    printf("Definindo parametros para a discretizao ... \n");

    // deviceCapabilities();

    printf("Alocando memOria no HOST ... \n");
    hostWave = (double *)malloc((N * N) * sizeof(double));       // Matrix Solution on HOST
    hostWaveFuture = (double *)malloc((N * N) * sizeof(double)); // Matrix Actualizations on HOST
    hostWavePast = (double *)malloc((N * N) * sizeof(double));   // Previous Matrix results on HOST

    printf("Inicializando matrizes no HOST ... \n");
    inicMatrix(hostWave);       // Zeros  Matrix
    inicMatrix(hostWaveFuture); // Zeros  Matrix
    inicMatrix(hostWavePast);   // Zeros  Matrix

    printf("Aplicando as condicoes iniciais a matriz ... \n");
    initialCond(hostWave); // Appling initial conditions

    writeFiles(hostWave, dx, dy);

    printf("Alocando memoria no DEVICE ... \n");
    CHECK(cudaMalloc(&deviceWave, (N * N) * sizeof(double)));       // Matrix Solution on Device
    CHECK(cudaMalloc(&deviceWaveFuture, (N * N) * sizeof(double))); // Matrix Actualizations on Device
    CHECK(cudaMalloc(&deviceWavePast, (N * N) * sizeof(double)));   // Matrix Actualizations on Device

    // deviceCapabilities();

    clock_t beginTime = clock();
    printf("Iniciando looping temporal ... \n");
    for (i = 0; i < IT; i++)
    {
        // printf("Transferindo informacoes do HOST para o DEVICE ... \n");
        CHECK(cudaMemcpy(deviceWave, hostWave, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(deviceWaveFuture, hostWaveFuture, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(deviceWavePast, hostWavePast, (N * N) * sizeof(double), cudaMemcpyHostToDevice));

        // Parellelism in N blocks and N threads per block for the inner elements
        waveEquationKernel<<<N, N>>>(deviceWave, deviceWaveFuture, deviceWavePast); // Parallelism in N Blocks with N Threads
        // actualizationKernel<<<N, N>>>(deviceWave, deviceWaveFuture, deviceWavePast);

        // printf("Transferindo atualizacoes do DEVICE para o HOST ... \n");
        CHECK(cudaMemcpy(hostWave, deviceWave, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostWaveFuture, deviceWaveFuture, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostWavePast, deviceWavePast, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));

        // printf("%f\t%f\t%f\n", hostWave[10], hostWaveFuture[10], hostWavePast[10]);

        for (k = 1; k < N - 1; k++)
        {
            for (j = 1; j < N - 1; j++)
            {
                hostWavePast[k * N + j] = hostWave[k * N + j];
                hostWave[k * N + j] = hostWaveFuture[k * N + j];
            }
        }
    }

    printf("Escrevendo o arquivo de dados ... \n");

    writeFiles(hostWave, dx, dy);

    clock_t endTime = clock();

    printf("Liberando memoria no HOST e no DEVICE ... \n");

    // Unallocing CPU variables
    free(hostWave);
    free(hostWaveFuture);
    free(hostWavePast);
    // Unallocing GPU variables
    cudaFree(deviceWave);
    cudaFree(deviceWaveFuture);
    cudaFree(deviceWavePast);

    printf("Time: %10.2f seconds \n", (endTime - beginTime) / (1.0 * CLOCKS_PER_SEC));
    return 0;
}
