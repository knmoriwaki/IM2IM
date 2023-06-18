#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <fitsio.h>

#include "para.h"
#include "proto.h"


int main(int argc, char *argv[])
{
	double mean_mean[2][NBIN_MEAN]={}, mean_var[2][NBIN_MEAN]={};
	double mesh[2][NBIN_MEAN_MESH][NBIN_MEAN_MESH]={};
    double smean, tmean;
	double xmax[2], xmin[2], dx[2], dx2[2];
	double *img;
	int i, j, iline, irun, count[2][NBIN_MEAN]={};

	char fname[256], sname[NLINE][256], true_dir[256], pred_dir[256], opt_name[256], odir[256], fout[256];

	fprintf(stderr, "### calc_mean.c ###\n");
	if( argc != 15 )
	{
		fprintf(stderr, "Usage: %s true_dir pred_dir opt_name odir sname1 sname2 irun_start nrun npix pixel_size xmin1 xmax1 xmin2 xmax2\n", argv[0]);
		fprintf(stderr, "Input: ");
		for(i = 0; i < argc; i++) fprintf(stderr, "%s ", argv[i]);
		fprintf(stderr, "\n");
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
	
	xmin[0] = atof( argv[11] );
	xmax[0] = atof( argv[12] );
	xmin[1] = atof( argv[13] );
	xmax[1] = atof( argv[14] );

	dx[0] = (xmax[0]-xmin[0])/(double)NBIN_MEAN;
	dx2[0] = (xmax[0]-xmin[0])/(double)NBIN_MEAN_MESH;
	dx[1] = (xmax[1]-xmin[1])/(double)NBIN_MEAN;
	dx2[1] = (xmax[1]-xmin[1])/(double)NBIN_MEAN_MESH;

	double norm = 1.0e-8;

	fprintf(stderr, "true_dir: %s\n", true_dir);
	fprintf(stderr, "pred_dir: %s\n", pred_dir);

	img = (double *) malloc(sizeof(double) * npix * npix);

	int ntot;
	for(ntot = 0, irun = irun_start; irun < irun_start + nrun; irun++)
	{
		for(iline = 0; iline < 2; iline++)
		{
			sprintf(fname, "%s/run%d_index0_%s.fits", true_dir, irun, sname[iline]);
			read_fits_file(fname, &img, npix);

			smean = 0.0;
			for(i = 0; i < npix*npix; i++)
			{
				smean += img[i] / norm / (double)( npix * npix );
			}
			
			sprintf(fname, "%s/gen_run%d_index0_%d.fits", pred_dir, irun, iline);
			read_fits_file(fname, &img, npix);

			tmean = 0.0;						
			for(i = 0; i < npix*npix; i++)
			{
				tmean += img[i] / norm / (double)( npix*npix );
			}
			
			int ibin = (int)( ( smean - xmin[iline] ) / dx[iline] );

			if( ibin >= 0 && ibin < NBIN_MEAN )
			{
				mean_mean[iline][ibin] += tmean;
				mean_var[iline][ibin] += tmean*tmean;
				count[iline][ibin] ++;
			}

			int ibin1 = (int)( ( smean - xmin[iline] ) / dx2[iline] );
			int ibin2 = (int)( ( tmean - xmin[iline] ) / dx2[iline] );
			if( ibin1 >=0 && ibin1 < NBIN_MEAN_MESH && ibin2 >= 0 && ibin2 < NBIN_MEAN_MESH )
			{
				mesh[iline][ibin1][ibin2] += 1.0;
			}
		}
		ntot ++;
		if( ((10*ntot)%nrun) == 0 ) fprintf(stderr, ".");
	}
	fprintf(stderr, "\n");

	for(iline = 0; iline < 2; iline++)
	{
		for(i = 0; i < NBIN_MEAN; i++)
		{
			mean_mean[iline][i] /= (double)count[iline][i];
			mean_var[iline][i] /= (double)count[iline][i];
			mean_var[iline][i] -= mean_mean[iline][i]*mean_mean[iline][i];
		}
	}

	// output //
	FILE *fp;
	for( iline = 0; iline < NLINE; iline++ )
	{
		// error bar //
		sprintf(fout, "%s/mean_%s_%s.txt", odir, sname[iline], opt_name);
		fp = fopen(fout, "w");
		fprintf(fp, "# xmin = %f\n", xmin[iline]);
		fprintf(fp, "# xmax = %f\n", xmax[iline]);
		for(i = 0; i < NBIN_MEAN; i++)
		{
			fprintf(fp, "%e %e %e\n"
					, xmin[iline] + dx[iline] * (0.5+i)
					, mean_mean[iline][i] 
					, sqrt( mean_var[iline][i] )
			);

		}
		fclose(fp);
		fprintf(stderr, "output: %s\n", fout);

		// mesh //
		sprintf(fout, "%s/mean_mesh_%s_%s.txt", odir, sname[iline], opt_name);
		fp = fopen(fout, "w");
		for(i = 0; i < NBIN_MEAN_MESH; i++)
		{
			for(j = 0; j < NBIN_MEAN_MESH; j++)
			{
				fprintf(fp, "%e ", mesh[iline][j][i]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		fprintf(stderr, "output: %s\n", fout);
	}
	
	free(img);

	return 0;
}

