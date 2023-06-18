#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <fftw3.h>
#include <fitsio.h>

#include "para.h"
#include "proto.h"

int main(int argc, char *argv[])
{
	double ps_mean[2][3][NBIN_PS]={}, ps_var[2][3][NBIN_PS]={}, k[NBIN_PS], ps[NBIN_PS], ps_tmp[NBIN_PS], ps_all[2][NMAX_PS][NBIN_PS];
	int i, j, iline, iout, irun;
	char fname[256], sname[NLINE][256], true_dir[256], pred_dir[256], opt_name[256], odir[256], fout[256];

	fprintf(stderr, "### calc_ps.c ###\n");
	if( argc != 11 )
	{
		fprintf(stderr, "Usage: %s true_dir pred_dir opt_name odir sname1 sname2 irun_start nrun npix pixel_size\n", argv[0]);
		exit(1);
	}

	sprintf(true_dir, "%s", argv[1]);
	sprintf(pred_dir, "%s", argv[2]);
	sprintf(opt_name, "%s", argv[3]);
	sprintf(odir, "%s", argv[4]);
	sprintf(sname[0], "%s", argv[5]);
	sprintf(sname[1], "%s", argv[6]);
	int irun_start = atoi( argv[7] );
	int nrun = atoi( argv[8] );
	int npix = atoi( argv[9] );
	double pixel_size = atof( argv[10] );

	double boxsize = pixel_size * npix; //arcmin
	double norm = 1.0;

	fprintf(stderr, "side length: %.4f deg\n", boxsize/60.);

	double *img_tmp;
	fftw_complex *img;
	img_tmp = (double *) malloc(sizeof(double) * npix * npix);
	img = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * npix * npix);

	int count;
	for(count = 0, irun = irun_start; irun < irun_start + nrun; irun++)
	{
		for(iline = 0; iline < 2; iline++)
		{
			for(iout = 0; iout < 2; iout++)
			{	
				if(iout == 0) sprintf(fname, "%s/run%d_index0_%s.fits", true_dir, irun, sname[iline]);
				if(iout == 1) sprintf(fname, "%s/gen_run%d_index0_%d.fits", pred_dir, irun, iline);
	
				read_fits_file(fname, &img_tmp, npix);
				//check(img_tmp, npix);
				for(i = 0; i < npix*npix; i++)
				{
					img[i][0] = img_tmp[i] / norm;
					img[i][1] = 0.0;
				}
				calc_auto_ps_2d(img, ps, npix, boxsize);
				
				for(i = 0; i < NBIN_PS; i++)
				{
					ps_mean[iline][iout][i] += ps[i] / nrun;
					ps_var[iline][iout][i] += ps[i] * ps[i] / nrun;
				}

				if( iout == 0 )
				{
					for(i = 0; i < NBIN_PS; i++)
					{
						ps_tmp[i] = ps[i];
					}
				}
				else
				{
					for(i = 0; i < NBIN_PS; i++)
					{
						ps_mean[iline][2][i] += (ps[i] - ps_tmp[i]) / nrun;
						ps_var[iline][2][i] += (ps[i] - ps_tmp[i]) * (ps[i] - ps_tmp[i]) / nrun;
						if( count < NMAX_PS ) ps_all[iline][count][i] = ps[i] - ps_tmp[i];
					}
				}
			}
			//if(iline==1) fprintf(stdout, "%e %e\n", ps_tmp[2], ps[2]);
		}
		count ++;
		if( count % (nrun/10) == 0 ) fprintf(stderr, ".");
	}
	fprintf(stderr, "\n");

	for(iline = 0; iline < 2; iline++)
	{
		for(iout = 0; iout < 3; iout++)
		{
			for(i = 0; i < NBIN_PS; i++)
			{
				ps_var[iline][iout][i] -= ps_mean[iline][iout][i]*ps_mean[iline][iout][i];
			}
		}
	}

	double sigma = 0.0;
	//sigma = 3.0 * 0.4; //arcmin
	// output //
	FILE *fp;
	for( iline = 0; iline < NLINE; iline++ )
	{
		sprintf(fout, "%s/ps_%s_%s.txt", odir, sname[iline], opt_name);
		fp = fopen(fout, "w");

		for(i = 0; i < NBIN_PS; i++)
		{
			k[i] = pow(10.0, logkmin + dlogk * (double)(i+0.5));
			double coef = 2.0 * M_PI * k[i] * k[i]; // / exp( - k[i]*k[i] * sigma*sigma );

			fprintf(fp, "%e %e %e %e %e %e %e\n"
					, k[i]
					, ps_mean[iline][0][i] * coef
					, sqrt( ps_var[iline][0][i] ) * coef
					, ps_mean[iline][1][i] * coef
					, sqrt( ps_var[iline][1][i] ) * coef
					, ps_mean[iline][2][i] * coef
					, sqrt( ps_var[iline][2][i] ) * coef
			);
		}
		fclose(fp);
		fprintf(stderr, "output: %s\n", fout);
	}

	
	// output power spectrum of all runs //
	/*
	FILE *fp;
	sprintf(fname, "ps_all.txt");
	fp = fopen(fname, "w");
	for(i = 0; i < NBIN_PS; i++)
	{
		k[i] = pow(10.0, logkmin + dlogk * (double)(i+0.5));
		coef = 2.0 * M_PI * k[i] * k[i];
		fprintf(fp, "%e ", k[i]);
		for(irun = 0; irun < NMAX_PS; irun++)
		{
			fprintf(fp, "%e %e "
					, ps_all[0][irun][i] * coef
					, ps_all[1][irun][i] * coef
				   );
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	*/
	
	fftw_free(img);
	free(img_tmp);

	return 0;
}
