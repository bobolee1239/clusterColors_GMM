#!/usr/bin/env python2
#
# ------------ Cluster Base Class -------------
# * Description:
#	  - A File to provide base class to inheritance to do clustering.
#
import numpy as np

## ------- K Means Base Class ----------
class Point(object):
	"""
	A base element in a cluster.
	"""
	def __init__(self, Attrs, normalizedAttrs = None):
		"""
		Argument:
			- Attrs: raw attribute of a point.
			- normalizedAttrs: normalized attributes of a point.
		"""
		self.originalAttrs = Attrs
		if normalizedAttrs is None:
			self.Attrs = Attrs
		else:
			self.Attrs = normalizedAttrs
	def dim(self):
		return len(self.Attrs)
	def getAttrs(self):
		return self.Attrs
	def getOriginalAttrs(self):
		return self.originalAttrs
	def distance(self, other):
		""" 
		Euclidean distance between 2 points. 
		"""
		vec1 = np.array(self.Attrs)
		vec2 = np.array(other.Attrs)
		return np.sqrt(sum((vec1 - vec2)**2))
	def toStr(self):
		return str(self.Attrs)
	def __str__(self):
		return self.toStr()

class Cluster(object):
	"""
	Container to collect couple of points.
	"""
	def __init__(self, points, pointType):
		self.points = points
		self.pointType = pointType
		self.centroid = self.computeCentroid()
		self.Cov = self.computeCovariance()
	
	def update(self, points):
		"""
		update cluster as points and return distance between new and old centroid.
		"""
		# record the old centroid 
		oldCentroid = self.centroid
		# set points to be points in cluster
		self.points = points
		# update new centroid
		if len(points) > 0:
			self.centroid = self.computeCentroid()
			self.Cov = self.computeCovariance()
			return oldCentroid.distance(self.centroid)
		else:
			return 0.0
	
	def members(self):
		"""
		Generator to generate points in this cluster.
		"""
		for p in self.points:
			yield p
	
	def computeCentroid(self):
		"""
		Return the self.pointtype of the mean.	 
		"""
		dim = self.points[0].dim()
		totVals = np.array([0.0] * dim)
		for p in self.points:
			totVals += np.array(p.getAttrs())
		totVals /= float(len(self.points))
		centroid = self.pointType(totVals, totVals)
		return centroid

	def computeCovariance(self):
		"""
		Compute covariance of memebers' attributes.
		"""
		dim = self.points[0].dim()
		Covariance = np.zeros((dim, dim))
		mean = np.array(self.centroid.getAttrs()).reshape(dim, 1)
		for p in self.points:
			attrs = np.array(p.getAttrs()).reshape(dim, 1)
			Covariance += (attrs - mean).dot((attrs - mean).T)
		Covariance /= float(len(self.points))
		return Covariance

	def getCentroid(self):
		return self.centroid
	def getCovariance(self):
		return self.Cov

class ClusterSet(object):
	"""
	Container to contain some clusters.
	"""
	def __init__(self, pointType):
		self.members = []
		self.pointType = pointType
	def add(self, c):
		"""
		c : a cluster
		"""
		if c in self.members:
			raise ValueError
		self.members.append(c)
	def getClusters(self):
		return self.members[:]
	def mergeClusters(self, c1, c2):
		"""
		c1 and c2 should be in self.members
		"""
		assert (c1 in self.members), 'Cluster is not in Cluster Set!'
		assert (c2 in self.members), 'Cluster is not in Cluster Set!'

		points = [] 		# list to record points in c1 and c2
		for p in c1.members():
			points.append(p)
		for p in c2.members():
			points.append(p)
		newC = Cluster(points, type(p))
		# remove old cluster
		self.members.remove(c1)
		self.members.remove(c2) 
		self.add(newC)

		return c1, c2

	def findClosest(self, metric):
		"""
		Arg:
		----------------------
		metric is the function you evaluate distance between points, eg. Euclidien.
		
		Return:
		----------------------
		Two closest cluster in cluster set to be merged.
		"""
		minDist = metric(self.member[0], self.member[1])
		toMerge = (self.member[0], self.member[1])
		for c1 in self.members:
			for c2 in self.members:
				if c1 == c2:
					continue
				if metric(c1, c2) < minDist:
					minDist = metric(c1, c2)
					toMerge = (c1, c2)
		return toMerge

	def mergeOne(self, metric, toPrint = False):
		"""
		Member Function to Merge Once.
		"""
		# Check if merging is neccessary
		if len(self.members) == 1:
			return None
		if len(self.members) == 2:
			return self.mergeClusters(self.members[0], self.members[1])

		toMerge = self.findClosest(metric)
		if toPrint:
			print 'Merging ...'
		return self.mergeClusters(toMerge[0], toMerge[1])

	def mergeN(self, metric, numCluster=1, history=[], toPrint = False):
		"""
		merget till N(numCluster) cluster inside the cluster set.
		
		Return:
		--------------------
		Merging history
		"""
		assert numCluster >= 1
		while len(self.members) > numClusters:
			merged = self.mergeOne(metric, toPrint)
			history.append(merged)					# Record the Merging history
		return history

	def numClusters(self):
		return len(self.members)

	
