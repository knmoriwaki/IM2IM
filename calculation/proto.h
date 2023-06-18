#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <fitsio.h>

#include "para.h"

extern void check(double *image, int Nmesh);
extern void read_fits_file(char fname[], double **image, int Nmesh);

extern void one_point_pdf(double *image, double pdf[], int Nmesh, double boxsize);
extern void one_point_pdf_log(double *image, double pdf[], int Nmesh, int boxsize);
extern void calc_auto_ps_2d(fftw_complex *deltar, double ps[], int Nmesh, double boxsize);
extern void calc_cross_ps_2d(fftw_complex *deltar1, fftw_complex *deltar2, double ps[], int Nmesh, double boxsize);
extern double n2(int ik, int Nmesh);
extern double n1(int ik, int Nmesh);
extern double window(int i1, int i2, int Nmesh);
extern void save_fits_image(float *image, char fname[], int npix);
extern void smooth_intensity_map(float **mesh_int, int npix_max, int npix_out, double smoothing_scale);

extern float median(int n, double x[]);
extern void peak2d(double *image, int index[][NMAX_PEAK], int *npeaks, int filter_size, double threshold, int Nmesh);
extern void gen_median_map(double **img_med, double *img, int Nmesh);
extern void set_threshold(double *image, int Nmesh, double *mean, double *threshold, double nsigma);
