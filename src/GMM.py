#!/usr/bin/env python2
#
## ----- GMM to Cluster Colors In JPEG File ------
#  * AUTHOR: Brian, Lee
#  * DATE: Dec 30th, 2017
#  * Description:
#		Using Kmeans to find means of each Gaussian
#		distribution quickly and then applied GMM to
#		cluster colors further.
## ------------------------------------------------
#
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from PIL import Image
from resizeimage import resizeimage

IN_FILE = '../SamplePics/img.jpg'
OUT_FILE = '../SamplePics/Resize.jpg'
KMEANS_FILE = '../KMeans_'
GMM_FILE = '../GMM_'
# ------ Kmeans Base class -------------
from KMeans import * 

## ------- JPG I/O --------
def Resize(infile, outfile, multiply=0.1):
	"""
	Resize input image and save it
	"""
	print 'Resizing Pic ...'
	im = Image.open(infile)
	rows, cols = im.size
	im = resizeimage.resize_cover(im, [rows*multiply, cols*multiply])
	im.save(outfile, im.format)
	im.close()

def readJPG(filename):
	"""
	Read *.jpg file and Return as array of Pixel(user defined data type).
	
	Return:
	-----------------------
	pixels : shape (rows, cols)
	"""
	im = Image.open(filename)
	pixs = im.load()

	rows, cols =  im.size
	pixels = []
	for r in xrange(rows):
		pixels.append([])
		for c in xrange(cols):
			# Normalized
			Attrs = []
			for e in pixs[r, c]:
				Attrs.append(e / 255.0)
			pixels[r].append(Pixel(pixs[r, c], Attrs))	
	
	pic_size = (rows, cols)
	
	im.close()
	return pixels, pic_size
	
def WriteJPG_KMeans(pixels, centroids, filename):
	"""
	Input pixels and Save as a jpeg file.
	"""
	print 'Writing ' + filename + ' ...'
	size = (len(pixels), len(pixels[0]))
	im = Image.new("RGB", size)
	pixs = im.load()
	
	# All Colors
	colors = []
	ColPix = []
	for c in centroids:
		rgb = []
		attrs = []
		for e in c:
			rgb.append(int(e * 255))
		colors.append(tuple(rgb))
		ColPix.append(Pixel(tuple(rgb), c))
	
	for r in xrange(size[0]):
		for c in xrange(size[1]):
			# find the closest color
			idx = None
			minDist = None
			for i in xrange(len(ColPix)):
				if (idx is None) or (minDist > pixels[r][c].distance(ColPix[i])):
					idx = i
					minDist = pixels[r][c].distance(ColPix[i])
			pixs[r, c] = colors[idx]

	im.save(filename)
	im.close()
	
	

def WriteJPG_GMM(pixels, Means, Covs, Priors, filename):
	"""
	Input pixels and Save as a jpeg file.
	"""
	print 'Writing ' + filename + ' ...'
	numAttrs = len(Means[0])
	size = (len(pixels), len(pixels[0]))
	im = Image.new("RGB", size)
	pixs = im.load()
	
	# All Colors
	colors = []
	ColPix = []
	for m in Means:
		rgb = []
		attrs = []
		for e in m:
			rgb.append(int(e * 255))
		colors.append(tuple(rgb))
		ColPix.append(Pixel(tuple(rgb), m))
	
#	for i in xrange(len(ColPix)):
#		print 'Mean #'+str(i), Means[i]
#		print 'Covs #'+str(i), Covs[i]
#		print 'Prios #'+str(i), Priors[i]

	for r in xrange(size[0]):
		for c in xrange(size[1]):
			# find the closest color
			Attrs = np.array(pixels[r][c].getAttrs())
			tot = 0.0
			idx = None
			maxPost = None
			for i in xrange(len(ColPix)):
				# Calculate Posterior
				Post = Priors[i] * (GaussianProb(Attrs, Means[i], Covs[i]))
				if (idx is None) or (maxPost < Post):
					idx = i
					maxPost = Post
			pixs[r, c] = colors[idx]

	im.save(filename)
	im.close()

## ------- Customer Data Type -----------
#  1.Inheriant from class Point in Kmeans.py
class Pixel(Point):  			
	"""
	Represent a small unit in a picture.
	"""
	def __init__(self, RGB, norRGB = None):
		"""
		"""
		Point.__init__(self, RGB, norRGB)


## ------ KMeans ---------
def KMeans(points, k, cutoff, pointType, maxIters = 100, toPrint = False):
	"""
	Arg:
	-------------------------
	- points <PointType> : points to be clustered.
	- k <int> : #cluster 
	- cutoff <float> : tolerance of error
	- pointType : type of points
	- maxIters <int> : max #Iteration to avoid infinte loop.
	- toPrint <bool> : to Print out Infomation.

	Returns:
	--------------------------
	- Clusters
	- Maximum Distance among clusters.
	"""
	# Get K random initial centroid among points
	initialCentroids = random.sample(points, k)
	clusters = []
	# Create a cluster for each centroid
	for p in initialCentroids:
		clusters.append(Cluster([p], pointType))
	
	numIters = 0
	biggestDist = cutoff
	# Iteration till close enough
	while (biggestDist >= cutoff) and (numIters < maxIters):
		numIters += 1
		if toPrint:
			print 'Iteration #' + str(numIters)
		# Create a list containing k empty list to store points
		newClusters = []
		for i in xrange(k):
			newClusters.append([])
		
		# Find the centroid closest to each point
		for p in points:
			minDist = p.distance(clusters[0].getCentroid())
			idx = 0
			for i in xrange(k):
				distance = p.distance(clusters[i].getCentroid())
				if distance < minDist:
					minDist = distance
					idx = i
			# Add Point to the cluster
			newClusters[idx].append(p)

		# Update each cluster and record the change
		biggestDist = 0.0
		for i in range(k):
			change = clusters[i].update(newClusters[i])
			biggestDist = max(change, biggestDist)
	
	if toPrint:
		maxDist = 0.0
		for c in clusters:
			for p in c.members():
				if p.distance(c.getCentroid()) > maxDist:
					maxDist = p.distance(c.getCentroid())
		print '# Iteration:', numIters, 'Max Diameter: ', maxDist
	
	return clusters

## ---------- GAUSSIAN MIXTURE MODEL ------------
# STRATEGY:
#	- EM Algorithm:
#		E STEP: Calculate the Posterior Probability
#		M STEP: ReCalculate the Mean, Covariance, and Prior
# -----------------------------------------------
def GaussianProb(x, mean, cov):
	"""
	"""
	if type(x) is not np.ndarray:
		x = np.array(x)
	if type(mean) is not np.ndarray:
		mean = np.array(mean)
	if type(cov) is not np.ndarray:
		cov = np.array(cov)
	
	dim = cov.shape[0]
	assert x.shape[0] == dim or x.shape[1] == dim
	assert mean.shape[0] == dim or mean.shape[1] == dim
	assert cov.shape == (dim, dim)

	# Check Singularity
	try:
		Precision = np.linalg.inv(cov)
	except np.linalg.LinAlgError as e:
		print '<WARN>', e
		cov = cov + np.eye(dim) * 0.00001
		Precision = np.linalg.inv(cov)
	
	pi = np.pi
	x = x.reshape(dim, 1)
	mean = mean.reshape(dim, 1)
	normalize = np.sqrt(np.linalg.det(cov) * ((2.0*pi)**dim)) 
	normalize = 1.0 / normalize
	
	density = np.exp(-0.5 * ((x - mean).T).dot(Precision).dot((x - mean)))
	return normalize * density[0][0]

def GMM(points, k, cutoff, pointType, initClusterSet = None, maxIters = 100, toPrint = False, toPlot = True):
	"""
	Arg:
	---------------------
	points <list of pointType> : points to be clustered
	k <int> : # of clusters 
	cutoff <float> : tolerance error 
	pointType : the class of each point
	maxIters <int> : to avoid infinite loop for some special initial condition.
	toPrint <bool> : Print out informations, just for Debugging.
	"""
	if initClusterSet is None:
		if toPrint:
			print 'No Initial Value is given, Doing KMeans First...'
		clusters = KMeans(points, k, cutoff, pointType, maxIters, toPrint)
	else:
		clusters = initClusterSet
		
	assert len(clusters) == k, 'K is not Equal to length of Initial Cluster Set!!'
	
	numPoints = len(points)
	numAttrs = len(points[0].getAttrs())
	# Init 
	singCovs = [] 		# record when the Covariance is Singular
	Priors = []				# shape = (k, )
	Means = []				# shape = (k, numAttrs)
	Covs = []				# shape = (k, numAttrs, numAttrs)
	for c in clusters:
		Priors.append(1.0/k)
		Means.append(c.getCentroid().getAttrs())
		Covs.append(c.getCovariance())
	
	Priors = np.array(Priors)
	Means = np.array(Means)
	Covs = np.array(Covs)
	Posteriors = np.zeros((len(points), k)) # shape = (numPoints, k)

	# Setup for Iteration
	movement = cutoff
	numIters = 0
	log_likelihood = np.zeros((maxIters, ))
	while movement >= cutoff and maxIters > numIters:
		numIters += 1
		if toPrint:
			print 'Iteration #' + str(numIters)

		# Create k empty list to store points in that cluster
		newClusters = []
		for i in xrange(k):
			newClusters.append([])

		# Effective Number of each Cluster
		effect_num = [0.0] * k

		# E Step : Calculate Posterior 
		totLikelihood = 0.0
		newMeans = np.zeros(Means.shape)
		for p in xrange(numPoints):
			tot = 0.0
			for i in xrange(k):
#				if toPrint:
#					print 'Posterior:'(multivariate_normal.pdf(np.array(points[p].getAttrs()), Means[i], Covs[i]))
				Posteriors[p][i] = Priors[i] * (GaussianProb(np.array(points[p].getAttrs()), Means[i], Covs[i]))
				tot += Posteriors[p][i]
			# Record log likelihood
			totLikelihood += np.log(tot)	
			for i in xrange(k):
				Posteriors[p][i] /= tot 
				# count effective number of each cluster here
				effect_num[i] += Posteriors[p][i]
				newMeans[i] += Posteriors[p][i] * np.array(points[p].getAttrs()).reshape(numAttrs, )
		
				
		
		# M Step : ReCalculate Means, Covs, Priors
		# 	- Re Clustering
		for p in xrange(numPoints):
			# find max Posterior 
			maxPost = Posteriors[p][0]
			maxIdx = 0
			for i in xrange(k):
				if (maxPost < Posteriors[p][i]):
					maxPost = Posteriors[p][i]
					maxIdx = i
			newClusters[i].append(points[p])

		# Update each cluster and record the change
		movement = 0.0
		newCovs = np.zeros(Covs.shape)
		for i in range(k):
			clusters[i].update(newClusters[i])
			# update Means
			oldMeans = Means[i]
			Means[i] = newMeans[i] / float(effect_num[i])
			change = np.sqrt(sum((Means[i] - oldMeans)**2))
			# update Covs
			for p in xrange(numPoints):
				attrs = np.array(points[p].getAttrs()).reshape(numAttrs, 1)
				mean = Means[i].reshape(numAttrs, 1)
				newCovs[i] += Posteriors[p][i] * (attrs - mean).dot((attrs - mean).T)
			
			# NOTE:
			#   - Adding a litte noise during training phase to aprrove robustness and 
			#     avoid singularity!! 
			Covs[i] = newCovs[i] / float(effect_num[i]) + np.eye(numAttrs) * 0.00001
		
			# Update Priors
			Priors[i] = effect_num[i] / float(numPoints)
			
			movement = max(change, movement)
#			if toPrint:
#				print 'new Means #' + str(i)+'\n', Means[i]
#				print 'new Covs #' + str(i)+'\n', Covs[i]
#				print 'new Priors #' + str(i)+'\n', Priors[i]
		# update Log Likelihood
		log_likelihood[numIters - 1] = totLikelihood


	if toPrint:
		maxDist = 0.0
		for c in clusters:
			for p in c.members():
				if p.distance(c.getCentroid()) > maxDist:
					maxDist = p.distance(c.getCentroid())
		print '# Iteration:', numIters, 'Max Diameter: ', maxDist
	if toPlot:
		plt.figure()
		plt.plot(range(1, len(log_likelihood) + 1), log_likelihood)
		plt.xlabel('# Iteration')
		plt.ylabel('Log Likelihood')
		plt.title('History of Likelihood for GMM, k = ' + str(k))
		plt.legend()
	
	return clusters, Means, Covs, Priors, singCovs
			
		
	


## ------ MAIN -----------
def main(numIters = 3, cutoff = 0.01, Print = False, saveJPG = True):
	Resize(IN_FILE, OUT_FILE)
	pixels, pic_size = readJPG(OUT_FILE) 		# pixels is a list of Pixel
	
	# Ravel()
	all_pixels = []
	for r in xrange(pic_size[0]):
		for c in xrange(pic_size[1]):
			all_pixels.append(pixels[r][c])
	 
	K = (2, 3, 5, 20)
	for k in K:
		# Using KMeans to Find Centroid
		print '----------------------------------'
		print 'Dealing with k =', k, '...'
		print 
		centroids = []
		for i in xrange(k):
			centroids.append([0.0] * 3)
		for i in xrange(numIters):
			clusters = KMeans(all_pixels, k, cutoff, pointType = Pixel, maxIters = 100, toPrint = Print)
			for c in xrange(k):
				for i in xrange(3):
					centroids[c][i] += (clusters[c].getCentroid()).getAttrs()[i]
		for c in xrange(k):
			for i in xrange(3):
				centroids[c][i] /= numIters

		if True:#Print:
			print 'For k =', k
			for c in xrange(k):
				print '\t-Centroid of cluster:', centroids[c]
		if saveJPG:
			WriteJPG_KMeans(pixels, centroids, KMEANS_FILE+str(k) + '.jpg')

		# < Initialize GMM Centroid as KMeans Result >
		# 	- Means' shape is (k, numAttrs)
		# 	- Covs' shape is (k, numAttrs, numAttrs)
		# 	- Priors' shape is (k, )
		clusters, Means, Covs, Priors, singCovs = GMM(all_pixels, k, 0, pointType = Pixel, 
													   initClusterSet = clusters, maxIters = 100, toPrint = False)
		#print Means
		#print Covs
		#print Priors
		singIter = ''
		for e in singCovs:
			singIter += (str(e)+', ')
		print '\t-Singular Covariance Occurs when iteration#:'
		print '\t' + singIter
		print 
		if saveJPG:
			WriteJPG_GMM(pixels, Means, Covs, Priors, GMM_FILE+str(k)+'.jpg')

	

if __name__ == '__main__':
	main()
	plt.show()
