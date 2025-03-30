import numpy as np
from PIL import Image
import cv2

class NormalizeImage(object):
	"""normalize image such as subtract mean, divide std"""

	def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
		if isinstance(scale, str):
			scale = eval(scale)
		self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
		mean = mean if mean is not None else [0.485, 0.456, 0.406]
		std = std if std is not None else [0.229, 0.224, 0.225]

		shape =  (1, 1, 3)
		self.mean = np.array(mean).reshape(shape).astype("float32")
		self.std = np.array(std).reshape(shape).astype("float32")

	def __call__(self, data):
		img = data["image"]
		if isinstance(img, Image.Image):
			img = np.array(img)
		assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
		
		img = (img.astype("float32") * self.scale - self.mean) / self.std
		img =  img.transpose((2, 0, 1))
		
		data['image'] = img
		return data
	
class Resize(object):
	def __init__(self, **kwargs):
		super(Resize, self).__init__()
		self.size = kwargs["size"]

	def __call__(self, data):
		
		img = data["image"]
		# When not keeping ratio
		img = cv2.resize(
			img,
			tuple(self.size)
		)
		data["image"] = img
		return data


class PrePrecessing(object):
	def __init__(self, config):
		super(PrePrecessing, self).__init__()
		self.transforms = []
		for transform_param in config:
			
			name= list(transform_param)[0]
			params = transform_param[name]
			
			self.transforms.append(eval(name)(**params))
	def __call__(self, data):
		for transform in self.transforms:
			data = transform(data)
		return data