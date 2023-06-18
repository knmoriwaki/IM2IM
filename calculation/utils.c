#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <fftw3.h>
#include <fitsio.h>

#include "para.h"
#include "proto.h"

void check(double *image, int Nmesh)
{
	int i;
	FILE *fp;
	char fname[256];

	sprintf(fname, "check.out");
	fp = fopen(fname, "w");

	for(i = 0; i < Nmesh*Nmesh; i++)
	{
		if(i%Nmesh == 0) fprintf(fp, "\n");
		fprintf(fp, "%e ", image[i]);
	}

	fclose(fp);
	fprintf(stderr, "output %s\n", fname);

}


void read_fits_file(char fname[], double **image, int Nmesh)
{
	int status = 0;
	fitsfile *fptr;
	char comment[FLEN_COMMENT];
	int bitpix = -64;
	int ng_x, ng_y;
	
	float nulval = 0.0;
	int anynull;
	long fpixel[] = {1,1};
	int ntot = Nmesh*Nmesh;

	if(fits_open_file(&fptr, fname, READONLY, &status))
	{
		fits_report_error(stderr, status);
		exit(1);
	}

	fits_read_key(fptr, TINT, "NAXIS1", &ng_x, comment, &status);
	fits_read_key(fptr, TINT, "NAXIS2", &ng_y, comment, &status);
	if(ng_x != Nmesh || ng_y != Nmesh)
	{
		fprintf(stderr, "Discrepancy between number of pixels: (%d, %d) and (%d, %d)!\n", ng_x, ng_y, Nmesh, Nmesh);
		exit(1);
	}

	fits_get_img_type(fptr, &bitpix, &status);
	fits_read_pix(fptr, TDOUBLE, fpixel, ntot, &nulval, *image, &anynull, &status);
	fits_close_file(fptr, &status);
}

void one_point_pdf(double *image, double pdf[], int Nmesh, double boxsize)
{
	int i, j;

	for(j = 0; j < NBIN_PDF; j++)
	{
		pdf[j] = 0.0;
	}

	for(i = 0; i < Nmesh*Nmesh; i++)
	{
		j = (int)( ( image[i] - INT_START ) / DINT );
		if( j >= 0 && j < NBIN_PDF )
		{
			pdf[j] += 1.0;
		}
	}
}

void one_point_pdf_log(double *image, double pdf[], int Nmesh, int boxsize)
{
	int i, j;

	for(j = 0; j < NBIN_PDF; j++) pdf[j] = 0.0;
	for(i = 0; i < Nmesh*Nmesh; i++)
	{
		j = (int)( ( log10( image[i] ) - LOGI_START ) / DLOGI );
		if( j >= 0 && j < NBIN_PDF )
		{
			pdf[j] += 1.0;
		}
	}
}
				

void calc_auto_ps_2d(fftw_complex *deltar, double ps[], int Nmesh, double boxsize)
{
	fftw_complex *deltak;
	fftw_plan plan;
	int cn2 = Nmesh/2 + 1;

	int i, ik[2], ips, imesh;
	double count[NBIN_PS], coef, re, im, temp, Nd = (double)( Nmesh );

	coef = boxsize * boxsize / ( Nd * Nd * Nd * Nd );
	
	deltak = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Nmesh * Nmesh);
	plan = fftw_plan_dft_2d(Nmesh, Nmesh, deltar, deltak, 1, FFTW_ESTIMATE);	
	fftw_execute(plan);
	
	for(i = 0; i < NBIN_PS; i++)
	{
		ps[i] = 0.0;
		count[i] = 0.0;
	}

	for(ik[0] = 0; ik[0] < Nmesh; ik[0]++){
	for(ik[1] = 0; ik[1] < Nmesh; ik[1]++){

		temp = sqrt( n2(ik[0],Nmesh) + n2(ik[1], Nmesh) );
		
		ips = (int)( ( log10( 2.0 * M_PI * temp / boxsize ) - logkmin ) / dlogk );
		imesh = ik[1] + ik[0]*Nmesh;

		if( ips < NBIN_PS && ips > -1 )
		{
			re = deltak[imesh][0];
			im = deltak[imesh][1];
			ps[ips] += re*re + im*im;
			count[ips] += 1.0;
		}
	}}

	for(i = 0; i < NBIN_PS; i++)
	{
		if( count[i] > 0 )
		{
			ps[i] = ps[i] / count[i] * coef;
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(deltak);
}



void calc_cross_ps_2d(fftw_complex *deltar1, fftw_complex *deltar2, double ps[], int Nmesh, double boxsize)
{
	fftw_complex *deltak1, *deltak2;
	fftw_plan plan1, plan2;
	int cn2 = Nmesh/2 + 1;

	int i, ik[2], ips, imesh;
	double count[NBIN_PS], coef, re, im, temp, Nd = (double)( Nmesh );

	coef = boxsize * boxsize / ( Nd * Nd * Nd * Nd );
	
	deltak1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Nmesh * Nmesh);
	deltak2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Nmesh * Nmesh);

	plan1 = fftw_plan_dft_2d(Nmesh, Nmesh, deltar1, deltak1, 1, FFTW_ESTIMATE);	
	plan2 = fftw_plan_dft_2d(Nmesh, Nmesh, deltar2, deltak2, 1, FFTW_ESTIMATE);	

	fftw_execute(plan1);
	fftw_execute(plan2);
	
	for(i = 0; i < NBIN_PS; i++)
	{
		ps[i] = 0.0;
		count[i] = 0.0;
	}

	for(ik[0] = 0; ik[0] < Nmesh; ik[0]++){
	for(ik[1] = 0; ik[1] < Nmesh; ik[1]++){

		temp = sqrt( n2(ik[0],Nmesh) + n2(ik[1], Nmesh) );
		
		ips = (int)( ( log10( 2.0 * M_PI * temp / boxsize ) - logkmin ) / dlogk );
		imesh = ik[1] + ik[0]*Nmesh;

		if( ips < NBIN_PS && ips > -1 )
		{
			ps[ips] += deltak1[imesh][0]*deltak2[imesh][0] + deltak1[imesh][1]*deltak2[imesh][1];
			count[ips] += 1.0;
		}
	}}

	for(i = 0; i < NBIN_PS; i++)
	{
		if( count[i] > 0 )
		{
			ps[i] = ps[i] / count[i] * coef;
		}
	}

	fftw_destroy_plan(plan1);
	fftw_destroy_plan(plan2);
	fftw_free(deltak1);
	fftw_free(deltak2);
}


double n2(int ik, int Nmesh)
{
	int cn2 = Nmesh / 2 + 1; 
	
	if(ik < cn2)
	{
		return (double)(ik * ik);
	}
	else
	{
		return (double)((Nmesh - ik) * (Nmesh - ik));
	}
}


double n1(int ik, int Nmesh)
{
	int cn2 = Nmesh / 2 + 1;
	if(ik < cn2)
	{
		return (double)(ik);
	}
	else
	{
		return (double)(Nmesh - ik);
	}
}

double window(int i1, int i2, int Nmesh)
{
	double x1, x2;

	if(i1 == 0 || i2 == 0)
	{
		return 1.0;
	}
	else
	{
		//x1 = M_PI * n1(i1, Nmesh) / (double)Nmesh;
		//x2 = M_PI * n2(i1, Nmesh) / (double)Nmesh;
		//
		x1 = M_PI * n1(i1, Nmesh) / (double)Nmesh;
		x2 = M_PI * n1(i2, Nmesh) / (double)Nmesh;

		return sin(x1)/x1 * sin(x2)/x2;
	}
}

void peak2d(double *image, int index[][NMAX_PEAK], int *npeaks, int filter_size, double threshold, int Nmesh)
{
	double local_max;
	int itemp, ii, jj, i, j, di, dj, imax, jmax;
	di = filter_size;
	dj = filter_size;

	int count = 0;

	for(i = 0; i < Nmesh; i++)
	{
		for(j = 0; j < Nmesh; j++)
		{
			if( count > NMAX_PEAK )
			{
				fprintf(stderr, "too small NMAX_PEAK\n");
				exit(1);
			}

			if( image[i*Nmesh+j] < threshold ) continue;

			local_max = 0.0;
			for(ii = -di; ii < di+1; ii++)
			{
				if( i+ii < 0 || i+ii >= Nmesh ) continue;
				for(jj = -di; jj < di+1; jj++)
				{
					if( jj*jj + ii*ii > di*di ) continue;
					if( j+jj < 0 || j+jj >= Nmesh ) continue;

					itemp = (i+ii)*Nmesh + (j+jj);
					if( image[itemp] > local_max)
					{
						local_max = image[itemp];
						imax = i+ii;
						jmax = j+jj;
					}
				}
			}

			if( i == imax && j == jmax )
			{
				index[0][count] = i;
				index[1][count] = j;
				count ++;	
			}
		}
	}

	*npeaks = count;
	//fprintf(stderr, "find %d peaks\n", count);
}


void save_fits_image(float *image, char fname[], int npix)
{
	long num[2];
	int status=0;
	long fpixel[]={1,1};
	int ntot;

	num[0]=npix;
	num[1]=npix;
	ntot=num[0]*num[1];

	fitsfile *fptr;

#ifdef OVERWRITE
	char tmp[256];
	sprintf(tmp, "%s", fname);
	sprintf(fname, "!%s", tmp); //see https://heasarc.gsfc.nasa.gov/docs/software/fitsio/quick/node7.html
#endif

	fits_create_file(&fptr,fname,&status);
	fits_create_img(fptr,-32,2,num,&status);

	fits_write_pix(fptr,TFLOAT,fpixel,ntot, image, &status);

	fits_write_key(fptr,TSTRING,"CTYPE1",(void*)"LINEAR","",&status);
	fits_write_key(fptr,TSTRING,"CTYPE2",(void*)"LINEAR","",&status);

	fits_close_file(fptr,&status);	
}

void make_gaussian_kernel(double w[], double sigma, int nw)
{
	int i, j;
	double wtot = 0.0;
	for(i = 0; i < NW*NW; i++) w[i] = 0.0;
	for(i = 0; i < nw; i++)
	{
		for(j = 0; j < nw; j++)
		{
			double r2 = (double)( i*i + j*j );
			if( r2 < nw*nw )
			{
				w[i+j*nw] = exp( -r2 / (2.0*sigma*sigma) );
				if( i == 0 && j == 0 ) wtot += w[i+j*nw];
				if( i == 0 && j > 0 ) wtot += 2*w[i+j*nw];
				if( i > 0 && j == 0 ) wtot += 2*w[i+j*nw];
				if( i > 0 && j > 0 ) wtot += 4*w[i+j*nw];
			}
		}
	}
	for(i = 0; i < nw*nw; i++) w[i] /= wtot;
}

void smooth_intensity_map(float **mesh_int, int npix_max, int npix_out, double smoothing_scale)
{
	float *temp;
	temp = (float *)malloc(sizeof(float) * npix_out*npix_out);

	double w[NW*NW];
	int nw;
	if( smoothing_scale < 0 )
	{
		nw = 1;
		w[0] = 1.0;	
	}
	else
	{
		nw = (int)( 2 * smoothing_scale + 1 );
		make_gaussian_kernel(w, smoothing_scale, nw);
	}

	for(int iz = 0; iz < 3; iz++)
	{
		for(int i = 0; i < npix_out*npix_out; i++) temp[i] = 0.0;

		for(int i = 0; i < npix_max; i++)
		{
			for(int j = 0; j < npix_max; j++)
			{
				double mtmp = (*mesh_int)[i+j*npix_max + iz*npix_max*npix_max];

		for(int ii = -nw+1; ii < nw; ii++)
		{
			for(int jj = -nw+1; jj < nw; jj++)
			{
				if( i + ii >= 0 && i + ii < npix_out && j + jj >= 0 && j + jj < npix_out )
				{
					int itmp = (i + ii) + (j + jj)*npix_out;
					int itmp_w = (int)( abs(ii) + abs(jj)*nw );
					temp[itmp] += mtmp * w[itmp_w];
				}
			}
		}
			}
		}

		for(int i = 0; i < npix_out*npix_out; i++) (*mesh_int)[i+iz*npix_max*npix_max] = temp[i];
	}


	free(temp);
}
