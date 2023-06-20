
true_dir=../val_data
opt_name=pix2pix_2_bs4_ep1_lambda1000_vanilla
#opt_name=pix2pix_2_bs4_ep1_lambda100_wgan
pred_dir=../output/${opt_name}/test

odir=./output

npix=256
pix_size=0.4

nrun=100
istart=0

opt=_sm3

sname1=z1.3_ha
sname2=z2.0_oiii

echo $opt_name

### basic statistics ###
./one_point_pdf $true_dir $pred_dir $opt_name $odir $sname1 $sname2 $istart $nrun $npix $pix_size
./calc_ps $true_dir $pred_dir $opt_name $odir $sname1 $sname2 $istart $nrun $npix $pix_size

### mean ### 
xmin1=0.75
xmax1=1.75
xmin2=0.55
xmax2=1.05
./calc_mean $true_dir $pred_dir $opt_name $odir $sname1 $sname2 $istart $nrun $npix $pix_size $xmin1 $xmax1 $xmin2 $xmax2

