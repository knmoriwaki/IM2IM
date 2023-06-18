true_dir=../val_data
opt_name=bs4_ep4_lambda100
pred_dir=../output/pix2pix_2_${opt_name}/test
odir=./output

npix=256
pix_size=0.4

nrun=100
istart=0


peak_th=2e-8
for peak_th in 0.5e-8 1e-8 2e-8 2.5e-8 3e-8 4e-8 4.5e-8 5e-8 5.5e-8 6e-8, 6.5e-8
do
	./peak_2d $true_dir $pred_dir $opt_name $odir 0 $istart $nrun $npix $pix_size $peak_th 
	./peak_2d $true_dir $pred_dir $opt_name $odir 1 $istart $nrun $npix $pix_size $peak_th 
done

