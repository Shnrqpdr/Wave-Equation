#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define mu 0.01
#define tension 40.0
#define N 600
#define xInicial 0
#define xFinal 600
#define yInicial 0
#define yFinal 600
#define finalTime 10000

void begin() {

    system("clear");
	printf("\n\n\n\n\t\t\t\t\t\tWelcome.");
	printf("\n\n\t\t Iniciando o programa...     ");

}

double ***allocArray() {

    int i, j;
    double ***waveFunction;

    waveFunction = (double***) malloc(N*sizeof(double));
    for(i = 0; i < N; i++){

        waveFunction[i] = (double**) malloc(N*sizeof(double));

        for(j = 0; j < N; j++) {
            
            waveFunction[i][j] = (double *) malloc(3*sizeof(double));

        }
    }

    return waveFunction;
}

double ***initialCondition(double ***wave, double dx, double dy){

	int i, j;

	for (i = 0; i < N; ++i){
		for (j = 0; j < N; ++j){

			wave[i][j][0] = sin(M_PI*i*dx/75);

		}
	}

	return wave;
}

double ***contourCondition(double ***wave, double dx, double dy){

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
	return wave;
}

double ***derivativeCondition(double ***wave, double dx, double dy){

    double v;
	int i, j;

    v = mu/tension;

	for (i = 1; i < N-1; i++)
	{
		for (j = 1; j < N-1; j++)
		{
			wave[i][j][1] = (2*wave[i][j][0]* + v*v*wave[i+1][j][0] + v*v*wave[i-1][j][0] + v*v*wave[i][j+1][0] +  v*v*wave[i][j-1][0])/2;
		}
	}

	return wave;
}

double ***finiteDifference(double ***wave, double dx, double dy){

    double v;

	int i, j, t;

    v = mu/tension;

	for (t = 1; t < finalTime; ++t) {
		for (i = 1; i < N-1; ++i) {
			for (j = 1; j < N-1; ++j) {
				wave[i][j][(t+1)%3] = 2*wave[i][j][t%3] - wave[i][j][(t-1)%3] + v*v*wave[i+1][j][(t%3)] + v*v*wave[i-1][j][t%3] + v*v*wave[i][j+1][t%3] +  v*v*wave[i][j-1][t%3];
			}
		}	
	}

    return wave;	
}

void writeFile(double ***wave, double dx, double dy) {

    int i, j, t;
    FILE *file;

    file = fopen("Wave.dat","w");

    fprintf(file, "x\ty\tt\tf\n");

    for (t = 1; t < finalTime; ++t) {
		for (i = 1; i < N-1; ++i) {
			for (j = 1; j < N-1; ++j) {
                fprintf(file, "%lf\t%lf\t%d\t%lf\n", i*dx, j*dy, (t+1)%3, wave[i][j][(t+1)%3]);
            }
        }
    }

}

void actionWork(double ***wave) {

    double dx, dy, dt;

    dx = (xFinal - xInicial)/N;
    dy = (yFinal - yInicial)/N;

    printf("Colocando condição inicial.\n");
    wave = initialCondition(wave, dx, dy);
    printf("Colocando condição de contorno.\n");
    wave = contourCondition(wave, dx, dy);
    printf("Colocando condição da derivada.\n");
    wave = derivativeCondition(wave, dx, dy);
    printf("Iniciando calculo da função de onda.\n");
    wave = finiteDifference(wave, dx, dy);
    printf("Escrevendo no arquivo");
    writeFile(wave, dx, dy);

}

void main() {

    double ***waveFunction;

    clock_t beginTime = clock();

    printf("Alocando a memória do array.\n");
    waveFunction = allocArray();

    printf("Começando os calculos.\n");
    actionWork(waveFunction);

    clock_t endTime = clock ();

    printf("Time: %10.2f seconds \n", (endTime - beginTime)/(1.0*CLOCKS_PER_SEC));
}