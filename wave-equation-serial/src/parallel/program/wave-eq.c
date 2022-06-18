#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define N 5000
#define xInicial 0
#define xFinal 500.0
#define yInicial 0
#define yFinal 500.0
#define finalTime 15000
#define alpha 0.4
#define gamma 0.8

double preencheArray(double wave[3][N][N], double valor)
{
    int i, j, t;

    for(t = 0; t < finalTime; t++){
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++){
                wave[t%3][i][j] = valor;
            }
        }
    }
    return wave[3][N][N];
}

double initialCondition(double wave[3][N][N], double dx, double dy)
{

    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {

            wave[0][i][j] = sin(M_PI * i / 75);
        }
    }

    return wave[3][N][N];
}

double contourCondition(double wave[3][N][N], double dx, double dy)
{

    int i, j, t;

    for (j = 0; j < N; j++)
    {
        for (t = 0; t < 3; t++)
        {
            wave[t][0][j] = 0;
            wave[N - 1][j][t] = 0;
        }
    }

    for (i = 0; i < N; i++)
    {
        for (t = 0; t < 3; t++)
        {
            wave[t][i][0] = 0;
            wave[t][i][N - 1] = 0;
        }
    }

    return wave[3][N][N];
}

double derivativeCondition(double ***wave, double dx, double dy)
{

    int i, j;

    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            wave[i][j][1] = (2 * wave[i][j][0] * (1 - alpha * alpha - gamma * gamma) + alpha * alpha * wave[i + 1][j][0] + alpha * alpha * wave[i - 1][j][0] + gamma * gamma * wave[i][j + 1][0] + gamma * gamma * wave[i][j - 1][0]) / 2;
        }
    }

    return wave[3][N][N];
}

void ***finiteDifference(double ***wave, double dx, double dy)
{

    int i, j, t;

    for (t = 1; t < finalTime; t++)
    {
        for (i = 1; i < N - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                wave[i][j][(t + 1) % 3] = 2 * wave[i][j][t % 3] * (1 - alpha * alpha - gamma * gamma) - wave[i][j][(t - 1) % 3] + alpha * alpha * wave[i + 1][j][(t % 3)] + alpha * alpha * wave[i - 1][j][t % 3] + gamma * gamma * wave[i][j + 1][t % 3] + gamma * gamma * wave[i][j - 1][t % 3];
            }
        }
    }
}

void writeFiles(double ***wave, double dx, double dy)
{

    int i, j, t;
    FILE *fileDynamicPlot, *fileStaticPlot;

    // fileDynamicPlot = fopen("Wave.dat","w");
    fileStaticPlot = fopen("WaveStatic.dat", "w");

    // fprintf(fileDynamicPlot, "x\ty\tt\tf\n");
    fprintf(fileStaticPlot, "x\ty\tt\tf\n");

    // for (t = 1; t < finalTime; t++) {
    // 	for (i = 1; i < N-1; i++) {
    // 		for (j = 1; j < N-1; j++) {
    //             fprintf(fileDynamicPlot, "%lf\t%lf\t%d\t%lf\n", i*dx, j*dy, t, wave[i][j][(t+1)%3]);
    //         }
    //     }
    // }

    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            fprintf(fileStaticPlot, "%lf\t%lf\t%d\t%lf\n", i * dx, j * dy, finalTime - 1, wave[i][j][(finalTime) % 3]);
        }
    }

    // fclose(fileDynamicPlot);
    fclose(fileStaticPlot);
}

void imprimeFinalTimeArray(double wave[3][N][N])
{

        for (int i = 0; i < nL; i++)
        {
            printf("\n");
            for (int j = 0; j < nC; j++)
            {
                printf("%lf \t", wave[finalTime % 3][i][j]);
            }
        }

        printf("\n\n");
}

// void actionWork(double ***wave)
// {

//     double dx, dy, dt;

    

//     printf("Colocando condição inicial.\n");
//     initialCondition(wave, dx, dy);

//     printf("Colocando condição de contorno.\n");
//     contourCondition(wave, dx, dy);

//     printf("Colocando condição da derivada.\n");
//     derivativeCondition(wave, dx, dy);

//     printf("Iniciando calculo da função de onda.\n");
//     finiteDifference(wave, dx, dy);

//     printf("Escrevendo no arquivo\n");
//     writeFiles(wave, dx, dy);
// }

void main(int argc, char **argv)
{
    int rank, size;
    int sinalMaster, sinalWorker;
    int i, j, k;
    int divisaoLinhas, aux, restoDivisao, divisaoMalha;
    int teste1, teste2;
    double waveFunction[3][N][N];
    double dx, dy, dt;

    dx = (xFinal - xInicial) / N;
    dy = (yFinal - yInicial) / N;
    printf("dx: %lf\tdy: %lf\n", dx, dy);
    printf("Malha: %d x %d\n", N, N);
    printf("Tempo total: %d\n", finalTime);

    clock_t beginTime = clock();

    sinalMaster = 0;
    sinalWorker = 1;
    restoDivisao = 0;
    divisaoMalha = 0;

    MPI_Status status;
    // inicia a zona paralela
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {

        printf("Alocando a memória do array.\n");
        waveFunction = preencheArray();
        waveFunction = initialCondition(waveFunction, dx, dy);
        waveFunction = contourCondition(waveFunction, dx, dy);
        waveFunction = derivativeCondition(waveFunction, dx, dy);

        imprimeFinalTimeArray(waveFunction);

        printf("Começando os calculos.\n");

    }

    // MPI_Barrier(MPI_COMM_WORLD);

    // teste1 = (N / size) * (rank);

    // teste2 = (N / size) * (rank + 1);

    // printf("Estou no rank %d e vou da linha %d até %d\n", rank, teste1, teste2);

    // printMatrix(waveFunction, teste1, teste2);

    if (rank == 0)
    {
        clock_t endTime = clock();

        printf("Time: %10.2f seconds \n", (endTime - beginTime) / (1.0 * CLOCKS_PER_SEC));
    }
}