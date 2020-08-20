"""
@author: 
Sandra Lorenz (Jakob)
s.lorenz@hzdr.de 
Helmholtz-Zentrum Dresden-Rossendorf
Helmholtz Institute Freiberg for Resource Technology
Chemnitzer Strasse 40
09599 Freiberg
Department of Exploration 
"""

"""
Definitions compilation for MEPHySTo (Mineral Exploration HyperSpectral Toolbox) - store together with MEPHySTo scripts or in your Python site-packages folder
"""

import re, os
import cv2
import numpy as np
import gdal


def findSIFT(arr2,contrastThreshold=0.01, sigma=1.0, edgeThreshold=10, EQUALIZE = 'False'):
    arr2=arr2.astype('float')
    gredgem = np.uint8((arr2-np.min(arr2))/(np.amax(arr2)-np.amin(arr2))*255)
    if EQUALIZE == 'True':
        gredgem = cv2.equalizeHist(gredgem)

    #  SIFT detector
    #print ('SIFT moving')
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrastThreshold, sigma=sigma, edgeThreshold=edgeThreshold)
    kp2sift=sift.detect(gredgem,None)
    kp2sift, des2sift=sift.compute(gredgem,kp2sift)
    #Matching points
    #print ('kpsift', len(kp2sift))
    return kp2sift,des2sift
def matchPointsCount(kp1,kp2,des1,des2,MatchDist=0.7, MIN_MATCH_COUNT = 5,algorithm=1, tree=5,check=100):
    index_params = dict(algorithm=algorithm, trees=tree)
    search_params = dict(checks=check)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    #print ('matches', len(matches))
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < MatchDist * n.distance:
            good.append(m)
    #print (len(good), ' good matches - band matched')
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return src_pts, dst_pts
    else:
        print ("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

def findORB(arr2,nfeatures=5000):
    arr2=arr2.astype('float')
    gredgem = np.uint8((arr2-np.min(arr2))/(np.amax(arr2)-np.amin(arr2))*255)
    #  SIFT detector
    #print 'SIFT moving'
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=nfeatures)
    kp2=orb.detect(gredgem,None)
    kp2, des2=orb.compute(gredgem,kp2)
    #Matching points
    #print ('kpsift', len(kp2))
    return kp2,des2
def matchPointsCountORB(kp1,kp2,des1,des2,MatchDist=0.7, MIN_MATCH_COUNT = 5,check=100):
    index_params = dict(algorithm=6,
                        table_number=8, # smaller is faster
                        key_size=6, #between about 6 and 14; 14 is faster but fails easier
                        multi_probe_level=2) #1 or 2; 2 is more accurate but way slower
    search_params = dict(checks=check)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    #print 'matches', len(matches)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < MatchDist * n.distance:
            good.append(m)
    #print (len(good), ' good matches - band matched')
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return src_pts, dst_pts
    else:
        print ("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
def NDVImask(img, wllist, NDVI_Threshold = 0.28):
    '''
    NDVI masking of spectral image
    :param img: spectral image
    :param: wavelength list
    :param NDVI_Threshold: NDVI threshold, pixel with higher NDVI values are masked, default = 0.28
    :return: masked image
    '''

    redwl=min(wllist, key=lambda x:abs(x-670))
    NIRwl=min(wllist, key=lambda x:abs(x-800))
    NIR=  wllist.index(NIRwl) +1             #800
    red=  wllist.index(redwl) +1             #670

    NDVI = np.squeeze(np.divide((img[:,:,NIR] - img[:,:,red]), (img[:,:,NIR] + img[:,:,red])))

    Mask=NDVI > NDVI_Threshold
    img[Mask] = 0

    return img

def DarkMask(img, Dark_Threshold = 0.0):
    '''
    Masking dark/shadowed areas of spectral image
    :param image: pectral image
    :param Dark_Threshold: Masking threshold in percent of the maximum mean spectral intensity.
    Pixel with lower values are masked; default = 0.0
    :return: masked image
    '''

    mean=np.mean(img,axis=2)
    max=np.squeeze(np.amax(mean))

    Mask=mean < max*Dark_Threshold

    img[Mask] = 0

    return img

def readHdrFile(hdrfilename):
    output = {}
    inblock = False

    try:
        hdrfile = open(hdrfilename, "r")
    except:
        print( "Could not open hdr file '" + str(hdrfilename) + "'")
        raise
    # end try

    # Read line, split it on equals, strip whitespace from resulting strings and add key/value pair to output
    currentline = hdrfile.readline()
    while (currentline != ""):
        # ENVI headers accept blocks bracketed by curly braces - check for these
        if (not inblock):
            # Split line on first equals sign
            if (re.search("=", currentline) != None):
                linesplit = re.split("=", currentline, 1)
                key = linesplit[0].strip()
                value = linesplit[1].strip()

                # If value starts with an open brace, it's the start of a block - strip the brace off and read the rest of the block
                if (re.match("{", value) != None):
                    inblock = True
                    value = re.sub("^{", "", value, 1)

                    # If value ends with a close brace it's the end of the block as well - strip the brace off
                    if (re.search("}$", value)):
                        inblock = False
                        value = re.sub("}$", "", value, 1)
                        # end if
                # end if
                value = value.strip()
                output[key] = value
                # end if
        else:
            # If we're in a block, just read the line, strip whitespace (and any closing brace ending the block) and add the whole thing
            value = currentline.strip()
            if (re.search("}$", value)):
                inblock = False
                value = re.sub("}$", "", value, 1)
                value = value.strip()
            # end if
            output[key] = output[key] + value
        # end if

        currentline = hdrfile.readline()
    # end while

    hdrfile.close()

    return output
def import_raster(filename,no_data=None, accuracy=np.float32):
    raster = gdal.Open(filename)                        # open image
    geoTransform = raster.GetGeoTransform()             # get geotransform (i.e. extend and pixel size)
    projection = raster.GetProjection()                 # get projection (i.e. UTM, WGS, etc)
    n_band = raster.RasterCount                         # get the number of bands
    image = {i:{} for i in range(1,n_band+1,1)}         # create an empty dictionary to store the bands
    for i in range(1,len(image)+1):                     # loop through each band, convert them to array and store them in 'image'
        data = raster.GetRasterBand(i).ReadAsArray()    # Get band 'i' and convert it to array
        data = data.astype(accuracy)                  # Change data format to float in order to remove no data values
        if no_data is not None:                         # Replace no data values with nan if argument is given
            data[data == no_data] = np.nan
        image[i] = data
    if n_band == 1:
        image = image[1]                                # if there is only one band drop off the dictionary structure
    return image, geoTransform, projection
