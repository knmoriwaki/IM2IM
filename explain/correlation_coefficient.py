import sys
import warnings
try:
    from multiprocessing import cpu_count
    THREADS = cpu_count()
    from pyfftw.interfaces.numpy_fft import fftn as _fftn, ifftn as _ifftn, fftfreq
    def fftn(*args, **kwargs):
        return _fftn(threads=THREADS, *args, **kwargs)
    def ifftn(*args, **kwargs):
        return _ifftn(threads=THREADS, *args, **kwargs)
except ImportError:
    #warnings.warn("You do not have pyFFTW installed. Installing it should give some speed increase.")
    print("Warning: you are not using FFTW.", file=sys.stderr)
    from numpy.fft import fftn, ifftn, fftfreq 
import numpy as np # import of numpy should be done after pyfftw import


def my_fft(X, L=None, b=1.): # b = 2 * np.pi for inverse FT

    dim = len(X.shape)
    N = np.array(X.shape)

    if L is None:
        L = N
    elif np.isscalar(L):
        L = L*np.ones(dim)

    dx = np.array([float(l)/float(n) for l, n in zip(L,N)]) # array of side length of cell
    Vx = np.product(dx) # volume of cell

    ### compute fft ###
    freq = np.array([fftfreq(n, d=d)*2*np.pi for n, d in zip(N, dx)]) ## k = 2pi * f
    ft = Vx * fftn(X) * np.sqrt(1/np.abs(b)) ** dim

    return ft, freq


def angular_average_nd(field, coords, nbins, n=None, log_bins=False, indx_min=0):
    ## can be used for real field only

    dim = len(field.shape)
    if len(coords) != dim:
        raise ValueError("coords should be a list of arrays, one for each dimensiont.")

    coords = np.sqrt(np.sum(np.meshgrid(*([x**2] for x in coords)), axis=0)) # k(i,j) = sqrt( ki*ki + kj*kj )

    ### define bins ###
    # "bins" here includes nbins + 1 components corresponding to the edge values of the k-bin
    # if you want to compute the "k of the bin", then you should take (bins[i]+bins[i+1])/2
    mx = coords.max()
    if not log_bins:
        bins = np.linspace(coords.min(), mx, nbins + 1)
    else:
        mn = coords[coords>0].min()
        bins = np.logspace(np.log10(mn), np.log10(mx), nbins + 1)

    ### set index for each pixel ###
    # label an index that satisfies bins[indx-1] <= coords < bins[indx]
    # indx takes 0, 1, ..., nbin+1
    indx = np.digitize(coords.flatten(), bins)

    ### compute the number of pixels in each bin ###
    # each component in bincount correaponds to the number of pixels with indx = 0, 1, ..., nbin+1
    # i.e., bincount has nbin + 2 = len(bins) + 1 components
    # the first component of bincount is for values < mn. 
    # the last component of bincount is for values > mx, where there should be no such pixels. So bincount[-1] is always 0.
    # we remove these two compontnes by setting [1:-1].
    sumweights = np.bincount(indx, minlength=len(bins)+1)[1:-1] 
    if np.any(sumweights==0):
        warnings.warn("One or more radial bins had no cell within it. Use a smaller nbins.")
    if np.any(sumweights==1):
        print("Warning: one or more radial bins have only one cell within it. This would result in inf in the variance")

    ### compute the mean in each bin ###
    # for each bin, sum up the field values, and then divide by the number of pixels 
    mean = np.bincount(indx, weights=np.real(field.flatten()), minlength=len(bins)+1)[1:-1] / sumweights

    ### compute variance ### 
    # for each bin, sum up the variance field values (i.e., (field-average)**2), and then divide by the number of pixels 
    average_field = np.concatenate(([0], mean, [0]))[indx]
    var_field = ( field.flatten() - average_field ) ** 2
    var = np.bincount(indx, weights=var_field, minlength=len(bins)+1)[1:-1] / (sumweights-1)

    return mean[indx_min:], bins[indx_min:], var[indx_min:]

def compute_power(deltax, deltax2=None, boxlength=1., nbins=20, log_bins=False):
    ## compmute 2d power spectrum 
    # deltax: input image np.array((N,N)) 
    # deltax2: second input image. If this is not None, the cross-power spectrum will be computed
    # boxlength: float or list of floats. The physical length of the side(s) in real space
    # nbins: int. number of bins

    if deltax2 is not None and deltax.shape != deltax2.shape:
        raise ValueError("deltax and deltax2 must have the same shpae!")

    dim = len(deltax.shape)

    if not np.iterable(boxlength):
        boxlength = [boxlength] * dim
    V = np.product(boxlength)

    # Fourier transformation
    FT, freq = my_fft(deltax, L=boxlength) 

    if deltax2 is not None:
        FT2, _ = my_fft(deltax2, L=boxlength)
    else:
        FT2 = FT

    # compute power spectrum
    P = np.real(FT * np.conj(FT2)) / V
    """ we need to implement the window function...
    wx = np.meshgrid(*([np.sin(k*l/2.)/(k*l/2.)] for k, l in zip(freq, boxlength))) ## array (dim, N, ..., N); each component ks[i] is the wavenumber map of i-th dimension
    window = np.ones(wx[0].shape)
    for w in wx:
        window = np.multiply(window, w)
    P = P / window**2
    """ 

    # compute angular power spectrum
    Pk, k, var = angular_average_nd(P, freq, nbins, log_bins=log_bins)

    return Pk, k, var


def compute_r(image1, image2, boxlength=1., nbins=20, log_bins=False):
    # compute cross-correlation coefficient between image1 and image2: r = Px / sqrt( P1 * P2 )
    Px, k, varx = compute_power(image1, deltax2=image2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    P1, _, var1 = compute_power(image1, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    P2, _, var2 = compute_power(image2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    return Px / np.sqrt( P1 * P2 ), k

