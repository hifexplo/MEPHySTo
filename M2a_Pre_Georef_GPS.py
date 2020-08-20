"""
Created on Tue 04.10.2016
Python script for automatic georeferencing/matching to orthophoto of *.hdr hyper- or multispectral image data.
To run the script first set input path and filename of image and orthophoto, then run.
Results will be stored within the input file folder.
@author: Sandra Jakob
"""

#IMAGE FLIPPED? SET/REMOVE MINUS IN line 178: src_coord_y.append(float64(-(GT[3] + GT[5] * src_pts[i,:,1])))
#_______________________________________________________________________________________________________________________
#       IMPORTS
#------------------------
import cv2
import HyDefinitions as HD
from scipy import *
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import glob, os
from osgeo import gdal
from gdalconst import *
from scipy.interpolate import *
from spectral import *
os.environ['GDAL_DATA'] = os.popen('gdal-config --datadir').read().rstrip()

#_______________________________________________________________________________________________________________________
#       DEFINITIONS
#------------------------
from osgeo import ogr
from osgeo import osr
def reprojectPoint(pointLat,pointLong, EPSGin, EPSGout):
    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(EPSGin)  # WGS84/Geographic
    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(EPSGout)  # WGS84 UTM Zone 56 South

    Point = ogr.Geometry(ogr.wkbPoint)
    Point.AddPoint(pointLat, pointLong)  # use your coordinates here
    Point.AssignSpatialReference(InSR)  # tell the point what coordinates it's in
    Point.TransformTo(OutSR)  # project it to the out spatial reference
    return Point.GetX(), Point.GetY()  # output projected X and Y coordinates

#_______________________________________________________________________________________________________________________
#       PARAMETERS
#------------------------
def Georef(MAIN, FILENAME, ORTHO,EPSG=25833,
    orthobands=3     ,
    HSIbands=10       ,
    SIFT_contrast1=0.01,
    SIGMA=1.5           ,
    MATCH_dist1=0.75     ,
    matching='poly'       ,
    RANSAC_thresh1=40      ,
    poly=3 ,
    Verbose=False           ):
    """
    Function for georeferencing of one or several spectral datacubes onto an Orthophoto
    :param MAIN: path to image folder
    :param FILENAME: single image name or '*.hdr' for all images in folder
    :param ORTHO: path to and name of Orthophoto
    :param EPSG: coordinate reference system key
    :param orthobands: amount of ortho bands used for matching
    :param HSIbands: stepwidth for HSI bands used for matching
    :param SIFT_contrast1: SIFT contrast threshold
    :param SIGMA: around 2.0 for small/sharp features, around 1.0 for big blurry ones
    :param MATCH_dist1: maximum matching distance: 0.75
    :param matching: choose transformation approach: 'poly' for polynomial approximation, 'grid' to forced adjustment to found GCP
    :param RANSAC_thresh1: RANSAC distance threshold: try 60 for poly, 40 for grid
    :param poly: only for 'poly': order of polynomial warping function
    :param Verbose:
    :return:
    """

    #_______________________________________________________________________________________________________________________
    #       MAIN - approximate matching
    #--------------------------------------


    ortho1, geoTransformortho, projection = HD.import_raster(ORTHO, no_data = None)

    os.chdir(MAIN)

    if not os.path.exists(MAIN+'georef/'):
         os.makedirs(MAIN+'georef/')
    for file in sorted(glob.glob(FILENAME)):
        try:
            print( file)

            filename = (MAIN+'georef/'+file.replace('.hdr', '_georef'))
            filenametemp = (MAIN +'georef/'+file.replace('.hdr','_temp'))

            hdtfile=MAIN[:-4]+file[:-7]+'.hdt'
            hdt = HD.readHdrFile(hdtfile)

            geo = hdt['geoposition'].strip('""')
            long = np.float(geo.split(' ')[0]) / 100
            lat = np.float(geo.split(' ')[2]) / 100
            Lat=(int(lat))+(int(100*(lat-int(lat))))/60.+(100*(lat-int(lat))-int(100*(lat-int(lat))))*6000/360000.0
            Long=(int(long))+(int(100*(long-int(long))))/60.+(100*(long-int(long))-int(100*(long-int(long))))*6000/360000.0
            X,Y=reprojectPoint(Long,Lat,4326,EPSG)
            minx = X-70
            miny = Y-20
            maxx = X+30
            maxy = Y+70

            ORTHOcut = ORTHO[:-4] + "_cut2"
            os.system('gdalwarp -of ENVI -overwrite -t_srs %s -te %s %s %s %s %s %s' % (
                'EPSG:' + str(EPSG), minx, miny, maxx, maxy, ORTHO,
            ORTHOcut))  # put -srcalpha if there is an alpha channel in your orthophoto
            orthocut, geoTransformortho, projection = HD.import_raster(ORTHOcut, no_data=None)
            head_ortho = HD.readHdrFile(ORTHOcut + '.hdr')

            kp1ortho = []
            des1ortho = []
            #for i in [2,4]:
            for i in range(1, orthobands):  # +1):
                try:
                    if orthocut.ndim == 2:
                        ortho1 = orthocut
                except:
                    ortho1 = orthocut[i]
                kp1f, des1f = HD.findSIFT(ortho1, contrastThreshold=0.01, sigma=SIGMA, edgeThreshold=30, EQUALIZE='False')
                for j in range(0, len(kp1f)):
                    kp1ortho.append(kp1f[j])
                    des1ortho.append(des1f[j])
            des1ortho = np.asarray(des1ortho)

            img, geoTransform2, projection2 = HD.import_raster(file[:-4], no_data=None)
            head=HD.readHdrFile(MAIN+file)
            #number of bands
            nbands=int(float("".join({head['bands']})))

            #Moving image
            kp1=[]
            des1=[]
            l = range(nbands)
            for i in (l[1::HSIbands]):

                f=img[i]
                arr1= f.astype(np.float)
                maxim1 = np.amax(arr1)
                minim1= np.amin(arr1)
                gredgem = np.uint8((arr1-minim1)/maxim1*255)
                kp1f, des1f = HD.findSIFT(gredgem, contrastThreshold=SIFT_contrast1, sigma=SIGMA, edgeThreshold=30,
                                          EQUALIZE='False')
                for j in range(0,len(kp1f)):
                    kp1.append(kp1f[j])
                    des1.append(des1f[j])
            des1=np.asarray(des1)

            #Matching points
            src_pts,dst_pts=HD.matchPointsCount(kp1ortho, kp1, des1ortho, des1, MatchDist=MATCH_dist1, MIN_MATCH_COUNT=5, algorithm=1, tree=5, check=100)

            M, mask=cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,RANSAC_thresh1)
            if matching== 'poly':
                dst_mask=dst_pts[:,0,:]*mask
                src_mask=src_pts[:,0,:]*mask
                dst_mask=dst_mask[dst_mask.all(1)]
                src_mask=src_mask[src_mask.all(1)]
                if Verbose:
                    plt.figure()
                    plt.imshow(gredgem, cmap='Greys_r')
                    plt.scatter(dst_mask[:,0],dst_mask[:,1])
                    plt.figure()
                    plt.imshow(ortho1, cmap='Greys_r')
                    plt.scatter(src_mask[:,0],src_mask[:,1])

                GT = float64(geoTransformortho)
                GT2 = float64(geoTransform2)
                src_coord_x=[]
                for i in range(0,len(src_pts)):
                    src_coord_x.append(float64(GT[0] + GT[1] * src_pts[i,:,0]))

                src_coord_y=[]
                for i in range(0,len(src_pts)):
                    src_coord_y.append(float64((GT[3] + GT[5] * src_pts[i,:,1])))
                src_coord=list(zip(src_coord_x, src_coord_y))

                dst_coord_x=[]
                for i in range(0,len(dst_pts)):
                    dst_coord_x.append(float64(GT2[0] + GT2[1] * dst_pts[i,:,0]))

                dst_coord_y=[]
                for i in range(0,len(src_pts)):
                    dst_coord_y.append(float64(GT2[3] + GT2[5] * dst_pts[i,:,1]))
                dst_coord=list(zip(dst_coord_x, dst_coord_y))

                GCP_list=np.concatenate((dst_coord,src_coord), axis=1)
                GCPmx= GCP_list*mask
                GCPlist=GCPmx[GCPmx.all(1)]


                seen=set()
                for i in range(0,GCPlist.shape[1]):
                    for j in range(0,GCPlist.shape[0]):
                        if j in seen:
                            GCPlist[i,:] = 0
                        seen.add(j)
                GCPlist=GCPlist[GCPlist.all(1)]
                #GCPlist=np.vstack((GCPlist, np.array([277.715, 104.073, 1975.47, -308.962]), np.array([398.944, 57.7549, 2576.74, -93.3742]),  np.array([503.589, 109.792, 3185.92, -441.479]), np.array([495.583, 65.1887, 3005.93, -150.732])))

                #Create GCPS
                GCP=''
                for i in range (0, GCPlist.shape[0]):
                    GCP=GCP+' -gcp '+str(GCPlist[i,0])+ ' ' +str(GCPlist[i,1])+ ' ' +str(GCPlist[i,2])+ ' ' +str(GCPlist[i,3])
                if len(GCPlist) > 6:

                    os.system('gdal_translate -of ENVI %s %s %s' %(GCP,file[:-4],filenametemp))
                    os.system('gdalwarp -order %s -of ENVI -overwrite -tr 0.08 0.08 -te %s %s %s %s -t_srs %s %s %s' % ( poly, minx, miny, maxx, maxy, 'EPSG:'+str(EPSG), filenametemp, filename))

                    os.remove(filenametemp)
                    os.remove(filenametemp+".hdr")
            elif matching=='grid':
                def writeHdrFile(filename, datatype, interleave="bsq"):
                    try:
                        hdrfile = open(filename, "w")
                    except:
                        print ("Could not open header file " + str(filename) + " for writing")
                        raise
                    # end try

                    hdrfile.write("ENVI\n")
                    hdrfile.write("description = \n")
                    hdrfile.write("samples = " + str(final.shape[1]) + "\n")
                    hdrfile.write("lines   = " + str(final.shape[0]) + "\n")
                    hdrfile.write("bands   = " + str(final.shape[2]) + "\n")
                    hdrfile.write("header offset = 0\n")
                    hdrfile.write("file type = ENVI Standard\n")
                    hdrfile.write("data type = " + str(datatype) + "\n")
                    hdrfile.write("interleave = " + interleave + "\n")
                    hdrfile.write("byte order = 0\n")
                    hdrfile.write("map info = {" + head_ortho['map info'] + "}\n")
                    hdrfile.write("coordinate system string = {" + head_ortho['coordinate system string'] + "}\n")
                    hdrfile.write("wavelength units= {" + head['wavelength units'] + "}\n")
                    hdrfile.write("wavelength= {" + head['wavelength'] + "}\n")
                    hdrfile.write("fwhm= {" + head['fwhm'] + "}\n")
                    hdrfile.flush()
                    hdrfile.close()
                # reshape source and destination points
                him = ortho1.shape[0]
                wim = ortho1.shape[1]
                grid_y, grid_x = np.mgrid[0:him, 0:wim]
                destination = np.uint16(dst_pts)
                destination = destination[:, 0, :]
                source = np.uint16(src_pts)
                source = source[:, 0, :]
                source = source * mask
                destination = destination * mask
                source = source[source.all(1)]
                destination = destination[destination.all(1)]
                seen = set()
                coord = np.concatenate((source, destination), axis=1)
                dst_mask = dst_pts[:, 0, :] * mask
                src_mask = src_pts[:, 0, :] * mask
                dst_mask = dst_mask[dst_mask.all(1)]
                src_mask = src_mask[src_mask.all(1)]
                if Verbose:
                    plt.figure()
                    plt.imshow(gredgem, cmap='Greys_r')
                    plt.scatter(dst_mask[:, 0], dst_mask[:, 1])
                    plt.figure()
                    plt.imshow(ortho1, cmap='Greys_r')
                    plt.scatter(src_mask[:, 0], src_mask[:, 1])
                # sort matches by x
                coord = coord[coord[:, 0].argsort()]
                print (len(coord))

                # delete redundant matches
                for i in range(0, coord.shape[1]):
                    for j in range(0, coord.shape[0]):
                        if j in seen:
                            coord[i, :] = 0
                        seen.add(j)
                coord = coord[coord.all(1)]
                print (len(coord))
                coord = np.float32(coord)
                coord_edge = coord
                coordlist = np.split(coord_edge, 2, axis=1)
                xlist = np.split(coordlist[0], 2, axis=1)
                ylist = np.split(coordlist[1], 2, axis=1)
                mapx = np.concatenate((coordlist[0], ylist[0]), axis=1)
                mapy = np.concatenate((coordlist[0], ylist[1]), axis=1)
                grid_z = griddata(coordlist[0], coordlist[1], (grid_x, grid_y), method='linear', fill_value=0)
                grid_z[:, :, 1] = np.where(grid_z[:, :, 0] == 0, 0, grid_z[:, :, 1])
                grid_z[:, :, 0] = np.where(grid_z[:, :, 1] == 0, 0, grid_z[:, :, 0])
                map_x = np.append([], [ar[:, 0] for ar in grid_z]).reshape(him, wim)
                map_y = np.append([], [ar[:, 1] for ar in grid_z]).reshape(him, wim)
                map_x_32 = map_x.astype('float32')
                map_y_32 = map_y.astype('float32')
                map_x_32 = np.where(map_x_32 < 0, 0, map_x_32)
                map_y_32 = np.where(map_y_32 < 0, 0, map_y_32)
                map_x_32 = np.where(map_x_32 > arr1.shape[1], 0, map_x_32)
                map_y_32 = np.where(map_y_32 > arr1.shape[0], 0, map_y_32)

                # loop for all bands, stack and save result
                List = []
                for i in range(0, nbands):
                    arr1 = img[(i + 1)]
                    warped = cv2.remap(arr1, map_x_32, map_y_32, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
                    List.append(warped)
                final = np.dstack(List)
                final = np.where(final == final[0, 0, :], 0, final)
                envi.save_image(filename + '_grid.hdr', final, interleave="bsq", ext="", force=True, metadata=head)

        except:
            print ("Could not georeference file", file)