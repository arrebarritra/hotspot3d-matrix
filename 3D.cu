#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BLOCK_SIZE 16
#define STR_SIZE 256

#define block_x_ 128
#define block_y_ 2
#define block_z_ 1
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#include "opt1.cu"

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016; float chip_width = 0.016; /* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void fatal(const char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
    int i,j,k;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
      fatal( "The file was not opened" );


    for (i=0; i <= grid_rows-1; i++)
      for (j=0; j <= grid_cols-1; j++)
        for (k=0; k <= layers-1; k++)
          {
            if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
            if (feof(fp))
              fatal("not enough lines in file");
            if ((sscanf(str, "%f", &val) != 1))
              fatal("invalid file format");
            vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
          }

    fclose(fp);

}


void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

    int i,j,k, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
      printf( "The file was not opened\n" );

    for (i=0; i < grid_rows; i++)
      for (j=0; j < grid_cols; j++)
        for (k=0; k < layers; k++)
          {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j+k*grid_rows*grid_cols]);
            fputs(str,fp);
            index++;
          }

    fclose(fp);
}

void computeTempCPU(float *pIn, float* tIn, float *tOut,
        int nx, int ny, int nz, float Cap,
        float Rx, float Ry, float Rz,
        float dt, int numiter)
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0;
    do{
        for(z = 0; z < nz; z++)
            for(y = 0; y < ny; y++)
                for(x = 0; x < nx; x++)
                {
                    c = x + y * nx + z * nx * ny;

                    w = (x == 0) ? c      : c - 1;
                    e = (x == nx - 1) ? c : c + 1;
                    n = (y == 0) ? c      : c - nx;
                    s = (y == ny - 1) ? c : c + nx;
                    b = (z == 0) ? c      : c - nx * ny;
                    t = (z == nz - 1) ? c : c + nx * ny;

                    tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp;
                }
        float *temp = tIn;
        tIn = tOut;
        tOut = temp;
        i++;
    }
    while(i < numiter);

}

float accuracy(float *arr1, float *arr2, int len)
{
    float err = 0.0;
    int i;
    for(i = 0; i < len; i++)
    {
        err += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
    }

    return (float)sqrt(err/len);
}


void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
    fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

    fprintf(stderr, "\t<iteration> - number of iterations\n");
    fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
    fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<outputFile - output file\n");
    exit(1);
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        usage(argc,argv);
    }

    char *pfile, *tfile, *ofile;
    int iterations = atoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    int numCols = atoi(argv[1]);
    int numRows = atoi(argv[1]);
    int layers = atoi(argv[2]);

    /* calculating parameters*/

    float dx = chip_height/numRows;
    float dy = chip_width/numCols;
    float dz = t_chip/layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;


    float *powerIn, *tempOut, *tempIn, *tempCopy;
    int size = numCols * numRows * layers;

    // Using cudaHostAlloc instead of calloc/malloc
    cudaError_t err;

    err = cudaHostAlloc((void**)&powerIn, size * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaHostAlloc((void**)&tempCopy, size * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaHostAlloc((void**)&tempIn, size * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaHostAlloc((void**)&tempOut, size * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* answer;
    err = cudaHostAlloc((void**)&answer, size * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize arrays to zero if needed
    memset(powerIn, 0, size * sizeof(float));
    memset(tempCopy, 0, size * sizeof(float));
    memset(tempIn, 0, size * sizeof(float));
    memset(tempOut, 0, size * sizeof(float));
    memset(answer, 0, size * sizeof(float));

    readinput(powerIn,numRows, numCols, layers,pfile);
    readinput(tempIn, numRows, numCols, layers, tfile);

    memcpy(tempCopy,tempIn, size * sizeof(float));

    hotspot_opt1(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);

    computeTempCPU(powerIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);

    float acc = accuracy(tempOut,answer,numRows*numCols*layers);
    printf("Accuracy: %e\n",acc);
    writeoutput(tempOut,numRows, numCols, layers, ofile);
    cudaFree(tempIn);
    cudaFree(tempOut);
    cudaFree(powerIn);
    return 0;
}


