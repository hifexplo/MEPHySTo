# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 15:39:44 2015
Python script for topographic correction of *.hdr hyper- or multispectral image data.
To run the script first set input paths of file and DEM, sunhorizon angle, azimuth and method, then run.
Results will be stored within the input file folder.
@author: Sandra Jakob
"""

import gdal
from gdalconst import *
import glob, os, math
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import Pysolar
import datetime
from spectral import *
from scipy import stats
from scipy import signal as signal
import scipy.ndimage.interpolation as sci
import HyDefinitions as HD


os.environ['GDAL_DATA'] = os.popen('gdal-config --datadir').read().rstrip()

def topo (filename, DEMw, TZ, DTw, Lat ,Lon , METHOD='cfactor',
		   si = [0, -1, 0, -1], sensorangle=0,EPSG='EPSG:32633', DEMband = 1):

	"""
	:param filename: path to spectral data
	:param DEMw: path to DEM
	:param METHOD: set method ('cosine', 'improvedcosine', 'gamma', 'percent', 'minnaert', 'minnaertslope', 'cfactor', 'se')
	:param si: for se & cfac: use image subset to calculate correction coefficients? if yes, then define si with upperleft x,y; lowerright x,y index of rectangular image subset
	:param sensorangle: if METHOD='gamma', set sensorangle (0 = nadir)
	:param EPSG: set project coordinate reference system code
	:param DEMband: 
	:param TZ: Time difference to UTC
	:param DT: (year, month, day, hour, minute, second, microsecond)
	:param Lat: Latitutde - positive in the northern hemisphere
	:param Lon: Longitude - negative reckoning west from prime meridian in Greenwich, England
	:return: 
	"""
	#=======================================================================================================================
	DT= datetime.datetime(DTw[0],DTw[1],DTw[2],DTw[3]+TZ,DTw[4],DTw[5],DTw[6])
	RIK= filename
	DEM= filename[:-4] +'_DEM'

	dat = envi.open(RIK)
	rik=dat.load()
	height, width, nbands= rik.shape
	raster = gdal.Open(RIK[:-4])  # open image
	gt = raster.GetGeoTransform()  # get geotransform (i.e. extend and pixel size)
	pj = raster.GetProjection()  # get projection (i.e. UTM, WGS, etc)

	head=dat.metadata

	minx = gt[0]
	miny = gt[3] + width*gt[4] + height*gt[5]
	maxx = gt[0] + width*gt[1] + height*gt[2]
	maxy = gt[3]

	os.system('gdalwarp -overwrite -t_srs %s -te %s %s %s %s %s %s' %(EPSG, minx, miny, maxx, maxy, DEMw, DEM))
	print ('calculating slope and aspect')
	Slope = RIK[:-4] +'_DEM_slope'
	Aspect = RIK[:-4] +'_DEM_aspect'

	os.system('gdaldem slope -b %s %s %s' %(DEMband, DEM, Slope))

	os.system('gdaldem aspect -zero_for_flat -b %s %s %s' %(DEMband, DEM, Aspect))


	AZIMUTH = 180 - Pysolar.GetAzimuth(Lat, Lon, DT)
	if AZIMUTH > 360:
		AZIMUTH = AZIMUTH - 360

	SUNHORIZONANGLE = Pysolar.GetAltitude(Lat, Lon, DT)

	h= SUNHORIZONANGLE		#sun horizon angle
	h_r=h*math.pi/180
	s=Image.open(Slope)		#terrain slope angle
	s= np.array(s.getdata()).reshape(s.size[::-1])
	s_r=s*math.pi/180
	o=Image.open(Aspect) 	#terrain aspect angle
	o=np.array(o.getdata()).reshape(o.size[::-1])
	o_r=o*math.pi/180
	z=90-h					#solar zenith angle
	z_r=z*math.pi/180
	a=AZIMUTH				#solar azimuth angle
	a_r=a*math.pi/180
	i=z-s					#incidence angle
	i_r=i*math.pi/180

	cos_i = (np.cos(s_r)) * (np.cos(z_r)) + (np.sin(s_r)) * (np.sin(z_r)) * (np.cos(a_r - o_r))



	cos_i = cv2.resize(cos_i, (width, height))
	cos_i = np.where(cos_i == 0, 0.00001, cos_i)
	s_r = cv2.resize(s_r, (width, height))


	List = []
	def cosine():
		print ("cosine")
		for i in range (0,nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			zi = np.divide(np.cos(z_r),cos_i)
			ref_c = r_o * zi
			List.append(ref_c)

	def improvedcosine():
		print ("improved cosine")
		cos_i_mean=np.mean(cos_i)
		for i in range (0,nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			ref_c = r_o + (r_o*(cos_i_mean - cos_i)/cos_i_mean)
			List.append(ref_c)

	def se():
		print ("se")
		for i in range(0, nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			cos_i_flat = cos_i[si[0]:si[1],si[2]:si[3]].flatten()
			r_o_flat = r_o[si[0]:si[1],si[2]:si[3]].flatten()

			Y = cos_i_flat
			X = r_o_flat
			X2 = X[X != 0]
			Y2 = Y[X != 0]
			m, b, r_value, p_value, std_err = stats.linregress(Y2, X2)
			ref_c = r_o -cos_i * m-b+np.mean(np.nonzero(r_o))
			List.append(ref_c)

	def cfactor():
		print ("c-factor")
		for i in range (0,nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			Y = cos_i[si[0]:si[1],si[2]:si[3]].flatten()
			X = r_o[si[0]:si[1],si[2]:si[3]].flatten()
			X2 = X[X != 0]
			Y2 = Y[X != 0]
			#plt.scatter(X2,Y2)
			slope, intercept, _, _, _ = stats.linregress(Y2, X2)
			c=intercept/slope
			List.append(r_o * np.divide(np.cos(z_r)+c, cos_i+c))

	def gamma(sensorangle):
		#sensorangle = 0 for nadir
		print ("gamma")
		for i in range (0,nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			viewterrain = math.pi/2-(sensorangle+s_r)
			ref_c = r_o * (np.cos(z_r) + np.cos(sensorangle)) / (cos_i+np.cos(viewterrain))
			List.append(ref_c)

	def minnaert():
		print ("minnaert")
		for i in range (0,nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			z = np.divide(cos_i,np.cos(z_r))
			z = np.where(z <= 0, 0.00001, z)
			with np.errstate(divide='ignore'):
				y = np.log(r_o)
				y[r_o <= 0] = 0
			x = np.log(z)
			s_f = s_r.flatten()
			y_f = y.flatten()
			x_f = x.flatten()
			#only use points where slope is >0.05
			y_f = y_f[(s_f > 0.05)]
			x_f = x_f[(s_f > 0.05)]
			#remove negative
			y_b = y_f[(x_f > 0) & (y_f > 0)]
			x_b = x_f[(x_f > 0) & (y_f > 0)]
			K, intercept, r_value, p_value, std_err = stats.linregress(x_b, y_b)
			z = np.divide(np.cos(z_r),cos_i)
			z=np.where(z < 0, 0, z)
			with np.errstate(divide='ignore'):
				ref_c = r_o * np.power(z, K)
				ref_c[z == 0] = 0
			List.append(ref_c)

	def minnaertslope():
		print ("minnaert with slope")
		for i in range (0,nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			with np.errstate(divide='ignore'):
				z = cos_i/(np.cos(z_r))
				# z[np.cos(z_r) == 0] = 0
			z[z <= 0] = 0.00001
			z=np.where(z == 0,0.00001, z)
			with np.errstate(divide='ignore'):
				y= np.log(r_o)
				y[r_o <= 0] = 0
			x= np.log(z)
			s_f=s_r.flatten()
			y_f=y.flatten()
			x_f=x.flatten()
			#only use points where slope is >5%
			y_f=y_f[(s_f>0.05)]
			x_f=x_f[(s_f>0.05)]
			#remove negative
			y_b=y_f[(x_f>0) & (y_f>0)]
			x_b=x_f[(x_f>0) & (y_f>0)]
			K, intercept, r_value, p_value, std_err = stats.linregress(x_b, y_b)
			z = np.divide(np.cos(z_r),cos_i*np.cos(s_r))
			z=np.where(z<0,0,z)
			with np.errstate(divide='ignore'):
				ref_c = r_o*np.cos(s_r) * np.power(z,(-K))
				ref_c[z == 0] = 0
			List.append(ref_c)

	def percent():
		print ("percent")
		for i in range(0, nbands):
			r_o=np.squeeze(np.nan_to_num(rik[:,:,i]))
			zi = np.divide(2.0,cos_i+1.0)
			ref_c = r_o * zi
			List.append(ref_c)

	if METHOD == 'cfactor':
		cfactor()
		methodname = 'cfac'
	elif METHOD == 'cosine':
		cosine()
		methodname = 'cos'
	elif METHOD == 'se':
		se()
		methodname = 'se'
	elif METHOD == 'improvedcosine':
		improvedcosine()
		methodname = 'cos2'
	elif METHOD == 'gamma':
		gamma(sensorangle)
		methodname = 'gam'
	elif METHOD == 'minnaert':
		minnaert()
		methodname = 'minn'
	elif METHOD == 'minnaertslope':
		minnaertslope()
		methodname = 'minslp'
	elif METHOD == 'percent':
		percent()
		methodname = 'perc'
	else:
		raise (ValueError, str(METHOD) + ' is no valid method')

	final = np.dstack(List)

	#perc=np.percentile(final, [99.99,0.01])
	#final=np.where(final<perc[1],perc[1],final)
	#final=np.where(final>perc[0],perc[0],final)

	savename = filename[:-4]+'_topocorr_'+methodname+".hdr"
	envi.save_image(savename, final, metadata=head, ext='', force=True, interleave="bsq")
	return final
