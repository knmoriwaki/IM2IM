import numpy as np
import astropy.io.fits as fits

def save_fits_data(image, path, norm=2.0e-7, overwrite=False):
    ## save astropy.io.fits 2d image data
    # image: np.array((1,1,N,N)) or torch.tensor((1,1,N,N)). The first (batch) and second (feature) dimensions will be squeezed.
    # path: output file name 
    # norm: set the same normalization factor used in load_data
    # overwrite: False in default
    img = image.squeeze()
    img = norm * img
    
    hdu = fits.PrimaryHDU(img)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=overwrite)
