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
	int i, j, irun, index_s[2][NMAX_PEAK], index_t[2][NMAX_PEAK], ns, nt, is, it;
	int true_count = 0, true_tmp, ns_tot = 0, nt_tot = 0;
	int filter_size = 10;
	double nsigma = 3.0;
	double mean1, mean2;

	char fname[256], sname[256], tname[256], true_dir[256], pred_dir[256], opt_name[256], odir[256], fout[256];
	
	fprintf(stderr, "### peak_2d.c ###\n");
	if( argc != 11 )
	{
		fprintf(stderr, "Usage: %s true_dir pred_dir opt_name odir map_id irun_start nrun npix pixel_size peak_th\n", argv[0]);
		exit(1);
	}

	sprintf(true_dir, "%s", argv[1]);
	sprintf(pred_dir, "%s", argv[2]);
	sprintf(opt_name, "%s", argv[3]);
	sprintf(odir, "%s", argv[4]);
	int map_id = atoi( argv[5] );
	int irun_start = atoi( argv[6] );
	int nrun = atoi( argv[7] );
	int Npix = atoi( argv[8] );
	double pixel_size = atof( argv[9] );
	double threshold = atof( argv[10] );

	if( map_id == 0 )
	{
		sprintf(sname, "z1.3_ha");
	}
	else if( map_id == 1 )
	{
		sprintf(sname, "z2.0_oiii");
	}
	else
	{
		fprintf(stderr, "Error: unknown map_id\n");
		exit(1);
	}

	double *img;
	img = (double *) malloc(sizeof(double) * Npix * Npix);
	if( img == NULL )
	{
		fprintf(stderr, "cannot allocate img\n");
		exit(1);
	}

	fprintf(stderr, "\n");
	fprintf(stderr, "pred_dir: %s\n", pred_dir);
	fprintf(stderr, "nrun = %d\n", nrun);
	fprintf(stderr, "filter size = %d\n", filter_size);
	fprintf(stderr, "threshold = %e\n", threshold);
	fprintf(stderr, "counting");

	sprintf(fname, "%s/run%d_index0_%s.fits", true_dir, 0, sname);
	read_fits_file(fname, &img, Npix);

	int count;
	for(count = 0, irun = irun_start; irun < irun_start+nrun; irun++)
	{
		// read true map //
		sprintf(fname, "%s/run%d_index0_%s.fits", true_dir, irun, sname);
		read_fits_file(fname, &img, Npix);
		peak2d(img, index_s, &ns, filter_size, threshold, Npix);

		// read reconstructed map //
		sprintf(fname, "%s/gen_run%d_index0_%d.fits", pred_dir, irun, map_id);
		read_fits_file(fname, &img, Npix);
		peak2d(img, index_t, &nt, filter_size, threshold, Npix);

		ns_tot += ns;
		nt_tot += nt;

		for( is = 0; is < ns; is ++)
		{
			for( it = 0; it < nt; it ++)
			{
				if( sqrt( pow(index_s[0][is] - index_t[0][it], 2.0 ) + pow(index_s[1][is] - index_t[1][it], 2.0 ) ) < filter_size )
				{
					true_count ++;
				}
			}
		}
		count ++;
		if( count%(nrun/10) == 0 ) fprintf(stderr, ".");
	}
	fprintf(stderr, "\n");

#ifdef SAVE_PEAK_FILE
	fprintf(stdout, "\n## %s ###\n", sname);
	fprintf(stdout, "# threshold = %e\n", threshold);
	fprintf(stdout, "# filter size = %d\n", filter_size);
	fprintf(stdout, "# npeak (true) = %d\n", ns_tot);
	fprintf(stdout, "# npeak (reconstructed) = %d\n", nt_tot);
	fprintf(stdout, "# true count = %d\n", true_count);
	fprintf(stdout, "# recall (true_count/npeak_true) = %.3f\n", (double)true_count/(double)ns_tot);
	fprintf(stdout, "# precision (true_count/npeak_reconstructed) = %.3f\n", (double)true_count/(double)nt_tot);
	fprintf(stdout, "####\n\n");
#else
	fprintf(stdout, "#nrun= %d\n", nrun);
	fprintf(stdout, "#filtersize= %d\n", filter_size);
	fprintf(stdout, "%e ", threshold);
	fprintf(stdout, "%f %f ", (double)true_count/(double)ns_tot, (double)true_count/(double)nt_tot);
	fprintf(stdout, "%d %d %d \n", true_count, ns_tot, nt_tot);
#endif

	free(img);

	return 0;
}

