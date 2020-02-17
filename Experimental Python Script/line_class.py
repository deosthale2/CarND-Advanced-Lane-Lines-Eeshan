import numpy as np

class Line(object):
	"""docstring for Line"""
	def __init__(self):
		self.detected = False
		self.base = []
		self.fit = []
		self.curverad = None
		self.allx = []
		self.ally = []
		self.diff_fit = np.array([0., 0., 0.])
