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
	double pdf_mean[NLINE][3][NBIN_PDF]={}, pdf_var[NLINE][3][NBIN_PDF]={}, pdf[NBIN_PDF], pdf_tmp[NBIN_PDF];
	double pdf_log_mean[NLINE][3][NBIN_PDF]={}, pdf_log_var[NLINE][3][NBIN_PDF]={}, pdf_log[NBIN_PDF], pdf_log_tmp[NBIN_PDF];
	int i, j, iline, iout, irun;
	char fname[256], sname[NLINE][256], true_dir[256], pred_dir[256], opt_name[256], odir[256], fout[256];
	
	fprintf(stderr, "### one_point_pdf.c ###\n");
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

	double *img;
	img = (double *) malloc(sizeof(double) * npix * npix);

	int count;
	for(count = 0, irun = irun_start; irun < irun_start + nrun; irun++)
	{
		for(iline = 0; iline < NLINE; iline++)// iline... 0: OIII, 1: Ha
		{
			for(iout = 0; iout < 2; iout++) //iout... 0: true, 1: predicted
			{	
				if(iout == 0) sprintf(fname, "%s/run%d_index0_%s.fits", true_dir, irun, sname[iline]);
				if(iout == 1) sprintf(fname, "%s/gen_run%d_index0_%d.fits", pred_dir, irun, iline);
	
				read_fits_file(fname, &img, npix);
				//check(img, npix);
				for(i = 0; i < npix*npix; i++)
				{
					img[i] /= norm;
				}
				
				one_point_pdf(img, pdf, npix, boxsize);
				one_point_pdf_log(img, pdf_log, npix, boxsize);
				
				for(i = 0; i < NBIN_PDF; i++)
				{
					pdf_mean[iline][iout][i] += pdf[i] / nrun;
					pdf_var[iline][iout][i] += pdf[i] * pdf[i] / nrun;

					pdf_log_mean[iline][iout][i] += pdf_log[i] / nrun;
					pdf_log_var[iline][iout][i] += pdf_log[i] * pdf_log[i] / nrun;
				}

				// deviation 
				if( iout == 0 )
				{
					for(i = 0; i < NBIN_PDF; i++)
					{
						pdf_tmp[i] = pdf[i];
						pdf_log_tmp[i] = pdf_log[i];
					}
				}
				else
				{
					for(i = 0; i < NBIN_PDF; i++)
					{
						pdf_mean[iline][2][i] += (pdf[i] - pdf_tmp[i]) / nrun;
						pdf_var[iline][2][i] += (pdf[i] - pdf_tmp[i]) * (pdf[i] - pdf_tmp[i]) / nrun;

						pdf_log_mean[iline][2][i] += (pdf_log[i] - pdf_log_tmp[i]) / nrun;
						pdf_log_var[iline][2][i] += (pdf_log[i] - pdf_log_tmp[i]) * (pdf_log[i] - pdf_log_tmp[i]) / nrun;
					}
				}
			}
			//if(iline==1) fprintf(stdout, "%e %e\n", pdf_tmp[2], pdf[2]);
		}
		count ++;
	}

	for(iline = 0; iline < NLINE; iline++)
	{
		for(iout = 0; iout < 3; iout++)
		{
			for(i = 0; i < NBIN_PDF; i++)
			{
				pdf_var[iline][iout][i] -= pdf_mean[iline][iout][i]*pdf_mean[iline][iout][i];
				pdf_log_var[iline][iout][i] -= pdf_log_mean[iline][iout][i]*pdf_log_mean[iline][iout][i];
			}
		}
	}

	// output //
	FILE *fp;
	for( iline = 0; iline < NLINE; iline++ )
	{
		sprintf(fout, "%s/pdf_%s_%s.txt", odir, sname[iline], opt_name);
		fp = fopen(fout, "w");
		for(i = 0; i < NBIN_PDF; i++)
		{
			double intensity = INT_START + DINT * (double)(i + 0.5);
			fprintf(fp, "%e ", intensity);
			for(j = 0; j < 3; j++)
			{
				fprintf(fp, "%e %e "
					, pdf_mean[iline][j][i]
					, sqrt( pdf_var[iline][j][i] ) 
					);
			}
			fprintf(fp, "\n");			
		}
		fclose(fp);
		fprintf(stderr, "output: %s\n", fout);

		sprintf(fout, "%s/pdf_log_%s_%s.txt", odir, sname[iline], opt_name);
		fp = fopen(fout, "w");
		for(i = 0; i < NBIN_PDF; i++)
		{
			double intensity = pow(10.0, LOGI_START + DLOGI * (double)(i + 0.5));
			fprintf(fp, "%e ", intensity);
			for(j = 0; j < 3; j++)
			{
				fprintf(fp, "%e %e "
					, pdf_log_mean[iline][j][i]
					, sqrt( pdf_log_var[iline][j][i] ) 
					);
			}
			fprintf(fp, "\n");			
		}
		fclose(fp);
		fprintf(stderr, "output: %s\n", fout);
	}

	free(img);

	return 0;
}

