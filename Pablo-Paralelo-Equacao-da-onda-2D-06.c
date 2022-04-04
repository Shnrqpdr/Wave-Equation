#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>



//Parametros
double	mu = 0.01;
double	tensao = 40;
double	xi = 0;
double	xf = 600;
double	yi = 0;
double	yf = 600;
double	ti = 0;
double	tf = 16000;
double	lambda = 0.4;
double	delta = 0.8;


void cabecalho(){
	//Cabecalho
	printf("\n");
	printf("====================================\n");
	printf("Programa Pablo-WaveEquation2D-06.c	v.06\n");
	printf("Autor: Pablo de Deus\n");
	printf("Data : 24/02/21\n");
	printf("Estado: Estavel\n");
	printf("====================================\n");
	printf("\n");
}

double ***alocando(int m, int n){
	int i, j;
	double ***z;
	//Alocando a matriz z
	z = (double***) malloc(n*sizeof(double));
	for (i = 0; i < n; ++i)
	{
		z[i]= (double**) malloc(m*sizeof(double));
		for (j = 0; j < m; ++j)
		{
			z[i][j] = (double*) malloc(3*sizeof(double));
		}
	}
	return z;
}

double ***alocandoInicial (double ***z, int m, int n, double dx, double dy){
	int i, j;
	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < m; ++j)
		{
			z[i][j][0] = sin(M_PI*i*dx/75);
		}
	}
	return z;
}

double ***contorno(double ***z, int m, int n){
	int i, j, k;

	//omp section
	//Em x = 0 e x = n
	for (j = 0; j < m; ++j)
	{
		for (k = 0; k < 3; ++k)
		{
			z[0][j][k] = 0;
			z[n-1][j][k] = 0;

		}
	}	

	//omp section
	//Em y = 0 e y = n
	for (i = 0; i < n; ++i)
	{
		for (k = 0; k < 3; ++k)
		{
			z[i][0][k] = 0;
			z[i][m-1][k] = 0;
		}
	}	
	return z;
}

double ***condicaodaderivada(double ***z, int m, int n){
	int i, j;
	for (i = 1; i < n-1; ++i)
	{
		for (j = 1; j < m-1; ++j)
		{
			z[i][j][1] = (2*z[i][j][0]*(1 - lambda*lambda - delta*delta) + lambda*lambda*z[i+1][j][0] + lambda*lambda*z[i-1][j][0] + delta*delta*z[i][j+1][0] +  delta*delta*z[i][j-1][0])/2;
		}
	}
	return z;
}

double ***calculoProximosElementos(double ***z, int m, int n, double dx, double dy, FILE *arquivo, int chunck){
	int i, j, k, p;
	p = (int)tf;
	for (k = 1; k < p; ++k)
	{
		#pragma omp parallel for default(shared) private (i, j) schedule(static, chunck)
		for (i = 1; i < n-1; ++i)
		{
			for (j = 1; j < m-1; ++j)
			{
				z[i][j][(k+1)%3] = 2*z[i][j][k%3]*(1 - lambda*lambda - delta*delta) - z[i][j][(k-1)%3] + lambda*lambda*z[i+1][j][(k%3)] + lambda*lambda*z[i-1][j][k%3] + delta*delta*z[i][j+1][k%3] +  delta*delta*z[i][j-1][k%3];
				
				//Salvando no arquivo a matriz quando t = tf-1
				if (k == tf-1)
				{
					fprintf(arquivo, "%lf\t%lf\t%lf\n",i*dx, j*dy ,z[i][j][(k+1)%3]);
				}
			}
		}	
	}	
}

int main(){
	double dx, dy;
	double ***z;
	int n, m, i, j, k, chunck;
	FILE *arquivo;
	chunck = 3;
	//Cabecalho
	cabecalho();

	//Comecando a contar o tempo
	clock_t begin = clock ();
 
	//Tamanho da malha
	n = 600;
	m = 600;

	//Calculando os elementos dx e dy
	dx = (xf-xi)/n;
	dy = (yf-yi)/m;

	//Alocando a matriz z
	z = alocando(m, n);

	//Colocando a condicao inicial
	z = alocandoInicial(z, m, n, dx, dy);

	//Colocando a condicao de contorno
	z = contorno(z, m, n);
	
	//Usando a condicao inicial da derivada em t = 0 para o calculo dos termos da matriz z para o tempo k = 1
	z = condicaodaderivada(z, m, n);

	//Abrindo o arquivo que sera salvo a matriz para um tempo especifico, t = tf-1
	arquivo = fopen("Wave.dat","w");

	//Calculo dos proximos termos da matriz z para k > 1
	z = calculoProximosElementos(z, m, n, dx, dy, arquivo, chunck);
	//Fechando o arquivo
	fclose(arquivo);

	//Terminando o tempo 
	clock_t end = clock ();

	printf("Tempo de execucao: %10.2f segundos \n", (end - begin)/(1.0*CLOCKS_PER_SEC));

	return 0;
}