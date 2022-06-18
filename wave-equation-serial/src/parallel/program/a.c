#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

// tamanho da dimensão da matriz
#define N 4

// função para imprimir uma matriz
double preencheMatriz(double matriz[N][N], int nL, int nC, double valor)
{

    if (matriz == NULL)
    {
        printf("Não foi possível persistir a matriz.\n");
    }
    else
    {
        for (int i = 0; i < nL; i++)
        {
            for (int j = 0; j < nC; j++)
            {
                if (i == j)
                {
                    matriz[i][j] = valor;
                }
            }
        }
    }

    return matriz[N][N];
}

void imprimeMatriz(double matriz[N][N], int nL, int nC)
{

    if (matriz == NULL)
    {
        printf("Não foi possível imprimir a matriz.\n");
    }
    else
    {

        for (int i = 0; i < nL; i++)
        {
            printf("\n");
            for (int j = 0; j < nC; j++)
            {
                printf("%lf \t", matriz[i][j]);
            }
        }

        printf("\n\n");
    }
}

int main(int argc, char **argv)
{
    // define as variáveis necessárias
    int rank, size;
    int sinalMaster, sinalWorker;
    double M1[N][N], M2[N][N], matrizProduto[N][N];
    int i, j, k;
    int divisaoLinhas, aux, restoDivisao, divisaoMalha;

    sinalMaster = 0;
    sinalWorker = 1;
    restoDivisao = 0;
    divisaoMalha = 0;

    MPI_Status status;

    srand((unsigned)time(NULL));

    // inicia a zona paralela
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // checa se o processo está sendo executado no core 0
    if (rank == 0)
    {
        // gera os valores_linhas
        M1[N][N] = preencheMatriz(M1, N, N, 2);
        M2[N][N] = preencheMatriz(M2, N, N, 3);

        // imprime as matrizes geradas
        // imprimeMatriz(M, N, N);
        // imprimeMatriz(N, N, N);

        // divide a malha (tamanho da matriz pelo número de cores disponíveis)
        divisaoMalha = N / (size - 1);

        // checa se o tamanho da matriz divide perfeitamente (restoDivisao 0) pelo número de cores ou não
        restoDivisao = N % (size - 1);

        //"marcador" do ponto inicial (vai definir o pedaço da matriz que cada core irá receber)
        aux = 0;

        for (i = 1; i < size; i++)
        {
            // se houver restoDivisao na divisão entre o tamanho da matriz pelo número de cores, distribui esse restoDivisao entre os cores
            if (i <= restoDivisao)
                divisaoLinhas = divisaoMalha + 1;
            else
                divisaoLinhas = divisaoMalha;

            // envia para os cores trabalhadores o "marcador" do ponto inicial de cada core, o número de linhas que eles vão utilizar, um pedaço de uma matriz e a outra matriz inteira
            MPI_Send(&aux, 1, MPI_DOUBLE, i, sinalMaster, MPI_COMM_WORLD);
            MPI_Send(&divisaoLinhas, 1, MPI_DOUBLE, i, sinalMaster, MPI_COMM_WORLD);
            MPI_Send(&M1[aux][0], divisaoLinhas * N, MPI_DOUBLE, i, sinalMaster, MPI_COMM_WORLD);
            MPI_Send(&M2, N * N, MPI_DOUBLE, i, sinalMaster, MPI_COMM_WORLD);

            // define o novo ponto do "marcador" após enviar para um core
            aux += divisaoLinhas;
        }

        for (i = 1; i < size; i++)
        {
            // recebe as informações dos trabalhores, no caso, o "marcador" do ponto inicial, o número de linhas que eles utilizaram e um pedaço da matriz resultante da multiplicação
            MPI_Recv(&aux, 1, MPI_DOUBLE, i, sinalWorker, MPI_COMM_WORLD, &status);
            MPI_Recv(&divisaoLinhas, 1, MPI_DOUBLE, i, sinalWorker, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrizProduto[aux][0], divisaoLinhas * N, MPI_DOUBLE, i, sinalWorker, MPI_COMM_WORLD, &status);
        }

        // imprime a matriz resultante
        // imprimeMatriz(P, N, N);
    }
    else
    {
        // recebe as informações adivindas do core mestre (0)
        MPI_Recv(&aux, 1, MPI_DOUBLE, 0, sinalMaster, MPI_COMM_WORLD, &status);
        MPI_Recv(&divisaoLinhas, 1, MPI_DOUBLE, 0, sinalMaster, MPI_COMM_WORLD, &status);
        MPI_Recv(&M1, divisaoLinhas * N, MPI_DOUBLE, 0, sinalMaster, MPI_COMM_WORLD, &status);
        MPI_Recv(&M2, N * N, MPI_DOUBLE, 0, sinalMaster, MPI_COMM_WORLD, &status);

        // realiza a multiplicação do pedaço da matriz recebida pela outra matriz
        for (j = 0; j < N; j++)
            for (i = 0; i < divisaoLinhas; i++)
            {
                matrizProduto[i][j] = 0;
                for (k = 0; k < N; k++)
                    matrizProduto[i][j] = matrizProduto[i][j] + M1[i][k] * M2[k][j];
            }

        // envia as informações para o core mestre
        MPI_Send(&aux, 1, MPI_DOUBLE, 0, sinalWorker, MPI_COMM_WORLD);
        MPI_Send(&divisaoLinhas, 1, MPI_DOUBLE, 0, sinalWorker, MPI_COMM_WORLD);
        MPI_Send(&matrizProduto, divisaoLinhas * N, MPI_DOUBLE, 0, sinalWorker, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
