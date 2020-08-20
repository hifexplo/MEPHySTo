"""
Created on 16.08.2018, last changes 16.01.2019
Python script for sensor-specific processing of a folder of
*.hdr RIKOLA hyperspectral image data. Point extraction using ORB: fast, but sometimes unstable.
To run the script first set main path containing the "CalibData" folder, then run.
Results will be stored in "MAIN/reg/".
@author: Sandra Lorenz, s.lorenz@hzdr.de

"""

import cv2
import numpy as np
import glob, os
import HyDefinitions as HD
from spectral import *



def Preprocess(MAIN, DistCorr = True, WARP = 'affine', match="ORB", ORBfeatures = 8000, MATCHDIST = 0.75, MIN_MATCH = 5,CONTTRESH = 0.03,
    EDGETHRESH = 10, SIGMA = 1.5):
    """
    Function to preprocess a folder of Senop Rikola datacubes. Results are saved in "MAIN/reg/"
    :param MAIN: Path to folder containing the "CalibData" folder
    :param DistCorr: Apply Distortion Correction according to camera parameters?
    :param WARP: Image transformation method: 'affine' or 'homography'
    :param match: choose "ORB" or "SIFT"
    :param ORBfeatures: Amount of extracted features using ORB
    :param MATCHDIST: Maximum point matching distance (higher for more but less accurate matches, usually between 0.65 and 0.8)
    :param MIN_MATCH:  Minimum amount of found good matches to proceed
    :param CONTTRESH: Contrast threshold for SIFT point detection (lower for more points, usually between 0.01 and 0.1)
    :param EDGETHRESH: Edge threshold for SIFT point detection
    :param SIGMA: Scaling factor for SIFT point detection (lower for smaller features, e.g., try 1.0, 1.5, 2.5)
    :return: Shows last processed image
    """
    #______________________________________________________________________________________________________________________________

    #CAMERA DISTORTION COEFFICIENTS:
    #camera matrix = ([[fx,skew,cx],[0,fy,cy],[0,0,1]])
    #distortion coefficients = ([[k1,k2,kp1,p2,k3]])


    # full Rikola image
    RIKMatfull = np.array([[1580, -0.37740987561726941, 532.14269389794072],[0,1586.5023476977308,552.87899983359080],[0,0,1]])
    RIKdistfull = np.array([[-0.34016504377397944, 0.15595251253428483, 0.00032919179911211665, 0.00016579344155373088,0.051315602289989909]])

    # half Rikola image
    RIKMathalf = np.array([[1580.9821817891338, -0.053468464819987738, 537.09531859948970],[0,1580.4094746112266,369.76442407125506],[0,0,1]])
    RIKdisthalf = np.array([[-0.31408677145500508, -0.26653256083139154, 0.00028155583639827883, 0.00025705469073531660, 2.4100464839836362]])


    if not os.path.exists(MAIN+'reg/'):
        os.makedirs(MAIN+'reg/')
    os.chdir(MAIN+"CalibData/")

    for file in glob.glob('*.hdr'):
        print("Processing "+file)
        data=envi.open(file, file[:-4]+'.dat')
        imag=data.load()
        head=data.metadata
        #output path
        filename = (MAIN+'reg/' + file.replace('.hdr','reg'))
        #number of bands
        nbands=int(head['bands'])
        lines = int(head['lines'])
        samples = int(head['samples'])

        try:
            #define first fixed image
            List=[]
            for i in range (0,nbands):
                z = imag[:,:,i]
                arr1 = z.astype(np.float)
                if DistCorr:
                    if (lines == 1010):
                        newcammat,roi=cv2.getOptimalNewCameraMatrix(RIKMatfull,RIKdistfull,(lines,samples),1,(lines,samples))
                        dstl=cv2.undistort(arr1,RIKMatfull,RIKdistfull,None, newcammat)
                        x,y,w,h=roi
                        dstl=dstl[y:y+h,x:x+w]
                        List.append(dstl)
                    else:
                        if (lines == 648):
                            newcammat,roi=cv2.getOptimalNewCameraMatrix(RIKMathalf,RIKdisthalf,(lines,samples),1,(lines,samples))
                            dstl=cv2.undistort(arr1,RIKMathalf,RIKdisthalf,None,newcammat)
                            x,y,w,h=roi
                            dstl=dstl[y:y+w,x:x+h]
                            List.append(dstl)
                        else:
                            raise (ValueError, "No calibration file exists for this image type - check number of lines")
                else:
                    List.append(arr1)

            img = np.dstack([List[0]]+List)
            im0 = img[:,:,0]
            arr1_zer= im0.astype(np.float)
            Listreg = [arr1_zer]
            j=0
            for i in range (0,nbands):
                j+=1
                # find characteristic points of image bands using ORB
                z = img[:,:,j]
                if match=="ORB":
                    kp2,des2= HD.findORB(z,nfeatures=ORBfeatures)
                else:
                    kp2, des2 = HD.findSIFT(z, contrastThreshold=CONTTRESH, sigma=SIGMA, edgeThreshold=EDGETHRESH)
                #print 'kp', len(kp2)
                for matchband in list(reversed(range(0,i+1))):
                    try:
                        base = Listreg[matchband]
                        if match == "ORB":
                            kp1,des1= HD.findORB(base,nfeatures=ORBfeatures)
                        else:
                            kp1, des1 = HD.findSIFT(base, contrastThreshold=CONTTRESH, sigma=SIGMA,
                                                edgeThreshold=EDGETHRESH)
                        #print 'kp1', len(kp1)

                        #Matching points
                        if match == "ORB":
                            src_pts,dst_pts=HD.matchPointsCountORB(kp1, kp2, des1, des2, MatchDist=MATCHDIST, MIN_MATCH_COUNT=MIN_MATCH, check=100)
                        else:
                            src_pts, dst_pts = HD.matchPointsCount(kp1, kp2, des1, des2, MatchDist=MATCHDIST,
                                                                   MIN_MATCH_COUNT=MIN_MATCH, algorithm=1, tree=5,
                                                                   check=100)
                        arr2 = z.astype('float')
                        if (WARP=='affine'):
                            H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)
                            dst_mask = dst_pts[:, 0, :] * status
                            src_mask = src_pts[:, 0, :] * status
                            dst_mask = dst_mask[dst_mask.all(1)]
                            src_mask = src_mask[src_mask.all(1)]
                            dst_mask=np.expand_dims(dst_mask, axis=1)
                            src_mask =np.expand_dims(src_mask, axis=1)
                            M=cv2.estimateRigidTransform(dst_mask,src_mask, False )
                            dst=cv2.warpAffine(arr2,M, (arr2.shape[1],arr2.shape[0]))
                        else:
                            H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                            dst = cv2.warpPerspective(arr2, H, (arr2.shape[1],arr2.shape[0]))

                        Listreg.append(dst)
                        break
                    except:
                        pass
                else:
                    print ("no match possible")
                    Listreg.append(z)
                    pass
                    #break
            Listreg=Listreg[1:]
            reg = np.dstack(Listreg)

            #CROPPING____________________________________________________________________________

            cropList = []
            null_list = []
            mask = np.ones(reg[:,:,0].shape).astype('uint8')*255
            for i in range (0,reg.shape[2]):
                b = reg[:,:,i]
                b= b.astype(np.float)
                nulls = np.where(b==0)
                mask[nulls]=0
            mask2 =mask/255
            thresh = mask

            _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt=contours[-1]
            x,y,w,h = cv2.boundingRect(cnt)
            x=x+5
            y=y+5
            w=w-20
            h=h-20

            for i in range (0,reg.shape[2]):
              cropb = reg[:,:,i]
              cropb= cropb.astype(np.float)
              cropb = mask2 * cropb
              crop=cropb[y:y+h,x:x+w]

              cropList.append(crop)
            final = np.dstack(cropList)

            #output data as ENVI hdr (bsq)_____________________________________________________
            envi.save_image(filename+".hdr", final, metadata=head, interleave="bsq", ext="",force=True)


        except:
            print ("Could not register file", file)

    imshow(final, bands=(10,20,30), stretch=0.05)