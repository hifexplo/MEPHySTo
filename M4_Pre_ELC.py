"""
Python script for radiometric correction of a folder of *.hdr RIKOLA hyperspectral image data using Empirical 
Line Calibration on refrence targets. 
First run the script until line 49 ("plt.imshow..."), in the plot define homogeneous rectangular ROIs on each 
calibration target by defining their upper left and lower right edges,
then proceed running the rest of the script. 

@author: Sandra Lorenz, s.lorenz@hzdr.de

"""
import numpy as np
import os,glob
from scipy import stats
from scipy.signal import savgol_filter
from spectral import *
def ELC(MAIN, FILENAME, wllist, black_panel=False, grey_panel=False, white_panel=False, SGF=False, SGF_width=7,
    SGF_poly=5):
    """
    Function for Empirical Line Correction
    :param MAIN: Path containing files to process
    :param FILENAME: #specify input image or write '*.hdr' to process all images in folder
    :param wllist: list of band wavelengths
    :param black_panel: image subset containing black reference pixels, False if not used
    :param grey_panel: image subset containing grey reference pixels, False if not used
    :param white_panel: image subset containing white reference pixels, False if not used
    :param SGF: Apply Savgol smoothing filter?
    :param SGF_width: width of smoothing window (only odd numbers)
    :param SGF_poly: degree of fitted polynom
    :return:
    """
    if not os.path.exists(MAIN + 'ELC/'):
        os.makedirs(MAIN + 'ELC/')
    lib_black = 'PVC_black.txt'
    lib_grey = 'PVC_grey.txt'
    lib_white = 'R90%_panel.txt'
    if isinstance(black_panel, np.ndarray):
        black_mean = np.median(black_panel, axis=(0,1))
        black_ref = np.loadtxt(lib_black, skiprows=1, dtype=np.float)
        black_ref[:,1]=black_ref[:,1]/100
        lib_wl_b = black_ref[:, 0].tolist()
        bands=len(black_mean)
    if isinstance(grey_panel, np.ndarray):
        grey_mean = np.median(grey_panel, axis=(0,1))
        grey_ref = np.loadtxt(lib_grey, skiprows=1, dtype=np.float)
        grey_ref[:, 1] = grey_ref[:, 1] / 100
        lib_wl_g = grey_ref[:, 0].tolist()
        bands = len(grey_mean)
    if isinstance(white_panel, np.ndarray):
        white_mean=np.median(white_panel, axis=(0,1))
        white_ref = np.loadtxt(lib_white, skiprows=0, dtype=np.float, delimiter=',')
        lib_wl_w = white_ref[:, 0].tolist()
        bands = len(white_mean)
    K_list=[]
    intercept_list=[]
    for i in range(0,bands):
        x=[]
        y=[]
        if isinstance(black_panel, np.ndarray):
            x.append(black_mean[i])
            y.append(black_ref[lib_wl_b.index(min(black_ref[:,0], key=lambda x:abs(x-wllist[i]))),1])
        if isinstance(grey_panel, np.ndarray):
            x.append(grey_mean[i])
            y.append(grey_ref[lib_wl_g.index(min(grey_ref[:,0], key=lambda x:abs(x-wllist[i]))),1])
        if isinstance(white_panel, np.ndarray):
            x.append(white_mean[i])
            y.append(white_ref[lib_wl_w.index(min(white_ref[:,0], key=lambda x:abs(x-wllist[i]))),1])
        if len(x)==1:
            x.append(0)
            y.append(0)
        K, intercept, r_value, p_value, std_err = stats.linregress(tuple(x),tuple(y))
        K_list.append(K)
        intercept_list.append((intercept))
    os.chdir(MAIN)
    for file in glob.glob(FILENAME):
        print(file)
        OUTPUT = MAIN + 'ELC/' + file[:-4] + '_ELC.hdr'
        dat= envi.open(MAIN+file)
        im=dat.load()
        for i in range(0, im.shape[2]):
            im[:,:,i]=im[:,:,i]*K_list[i] + intercept_list[i]
        if SGF:
            im=savgol_filter(im, SGF_width, SGF_poly, axis=2)
        envi.save_image(OUTPUT, im, Force=True, ext="", interleave="bsq", metadata=im.metadata)
    imshow(im, bands=(10,20,30))

