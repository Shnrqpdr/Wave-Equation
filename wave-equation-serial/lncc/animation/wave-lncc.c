#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 100
#define xInicial 0
#define xFinal 50.0
#define yInicial 0
#define yFinal 50.0
#define finalTime 300
#define	lambda 0.4
#define	delta 0.8

double ***allocArray() {

    int i, j;
    double ***waveFunction;

    waveFunction = (double***) malloc(N*sizeof(double));
    for(i = 0; i < N; i++){

        waveFunction[i] = (double**) malloc(N*sizeof(double));

        for(j = 0; j < N; j++) {
            
            waveFunction[i][j] = (double *) malloc(finalTime*sizeof(double));

        }
    }

    return waveFunction;
}

void ***initialCondition(double ***wave, double dx, double dy){

	int i, j;

	for (i = 0; i < N; ++i){
		for (j = 0; j < N; ++j){

			wave[i][j][0] = 4*sin(M_PI*i/75);

		}
	}
}

void ***contourCondition(double ***wave, double dx, double dy){

	int i, j, t;

	for (j = 0; j < N; j++)
	{
		for (t = 0; t < 3; t++)
		{
			wave[0][j][t] = 0;
			wave[N-1][j][t] = 0;

		}
	}	

	for (i = 0; i < N; i++)
	{
		for (t = 0; t < 3; t++)
		{
			wave[i][0][t] = 0;
			wave[i][N-1][t] = 0;
		}
	}	
}

void ***derivativeCondition(double ***wave, double dx, double dy){

    double v;
	int i, j;

	for (i = 1; i < N-1; i++)
	{
		for (j = 1; j < N-1; j++)
		{
			wave[i][j][1] = (2*wave[i][j][0]*(1 - lambda*lambda - delta*delta) + lambda*lambda*wave[i+1][j][0] + lambda*lambda*wave[i-1][j][0] + delta*delta*wave[i][j+1][0] +  delta*delta*wave[i][j-1][0])/2;
		}
	}
}

void ***finiteDifference(double ***wave, double dx, double dy){

    double v;

	int i, j, t;

	for (t = 1; t < finalTime; ++t) {
		for (i = 1; i < N-1; ++i) {
			for (j = 1; j < N-1; ++j) {
				wave[i][j][t+1] = 2*wave[i][j][t]*(1 - lambda*lambda - delta*delta) - wave[i][j][t-1] + lambda*lambda*wave[i+1][j][(t)] + lambda*lambda*wave[i-1][j][t] + delta*delta*wave[i][j+1][t] +  delta*delta*wave[i][j-1][t];
			}
		}	
	}
}

void writeFiles(double ***wave, double dx, double dy) {

    int i, j, t;
    FILE *fileDynamicPlot, *fileStaticPlot;

     fileDynamicPlot = fopen("Wave.dat","w");
    fileStaticPlot = fopen("WaveStatic.dat", "w");

     fprintf(fileDynamicPlot, "x\ty\tt\tf\n");
    fprintf(fileStaticPlot, "x\ty\tt\tf\n");

     for (t = 0; t < finalTime; ++t) {
	 	for (i = 1; i < N-1; ++i) {
	 		for (j = 1; j < N-1; ++j) {
                 fprintf(fileDynamicPlot, "%lf\t%lf\t%d\t%lf\n", i*dx, j*dy, t, wave[i][j][t]);
             }
         }
     }

    for (i = 1; i < N-1; ++i) {
		for (j = 1; j < N-1; ++j) {
            fprintf(fileStaticPlot, "%lf\t%lf\t%d\t%lf\n", i*dx, j*dy, finalTime - 1, wave[i][j][finalTime - 1]);
        }
    }

    fclose(fileDynamicPlot);
    fclose(fileStaticPlot);
}

void actionWork(double ***wave) {

    double dx, dy, dt;

    dx = (xFinal - xInicial)/N;
    dy = (yFinal - yInicial)/N;
    printf("dx: %lf\tdy: %lf\n", dx, dy);
    printf("Malha: %d x %d\n", N, N);
    printf("Tempo total: %d\n", finalTime);

    printf("Colocando condição inicial.\n");
    initialCondition(wave, dx, dy);

    printf("Colocando condição de contorno.\n");
    contourCondition(wave, dx, dy);

    printf("Colocando condição da derivada.\n");
    derivativeCondition(wave, dx, dy);

    printf("Iniciando calculo da função de onda.\n");
    finiteDifference(wave, dx, dy);

    printf("Escrevendo no arquivo\n");
    writeFiles(wave, dx, dy);

}

void main() {

    double ***waveFunction;

    //system("clear");

    clock_t beginTime = clock();

    printf("Alocando a memória do array.\n");
    waveFunction = allocArray();

    printf("Começando os calculos.\n");
    actionWork(waveFunction);

    clock_t endTime = clock ();

    printf("Time: %10.2f seconds \n", (endTime - beginTime)/(1.0*CLOCKS_PER_SEC));
}