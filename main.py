
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import load_config,load_model,visualize
from processing.postprocessing import DBPostProcess
from processing.preprocessing import PrePrecessing
from model_layers.resnet import ResNet
from model_layers.dbfpn_neck import DBFPN
from model_layers.db_head import DBHead
from paddle import nn
import cv2
import paddle

class Model(nn.Layer):
	def __init__(self, config_file):
		super(Model, self).__init__()
		
		self.config = load_config(config_file)
		self.model_config = self.config['Architecture']
		
		#build backbone
		self.backbone = ResNet(**self.model_config["Backbone"])
		in_channels = self.backbone.out_channels
		#build neck
		self.model_config["Neck"]["in_channels"] = in_channels
		self.neck = DBFPN(**self.model_config["Neck"])
		in_channels = self.neck.out_channels
		#build head
		self.model_config["Head"]["in_channels"] = in_channels
		self.head = DBHead(**self.model_config["Head"])
		#load_pretrained model
		load_model(self.config,self)
		
		#build preprocessing
		preprocessing_config = self.config['Preprocessing']
		self.preprocessing = PrePrecessing(preprocessing_config)
		
		#build post processing
		post_processing_config = self.config['PostProcess']
		self.post_processing = DBPostProcess(**post_processing_config)

		self.eval()
	def forward(self, x):
		x = self.backbone(x)
		x = self.neck(x)
		x = self.head(x)
		return x

	def infer(self,x,shape_list = None):
		with paddle.no_grad():
			if shape_list is None:
				shape_list = [x.shape[:2] for x in x]
			x = self.forward(x)
			return self.post_processing(x,shape_list)
	def load_image(self,image):
		if isinstance(image,str):
			image = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB)
		org_h,org_w = image.shape[:2]
		data  = {
			'image': image
		}
		data = self.preprocessing(data)
		image = data['image']
		
		return image,(org_h,org_w)
	def infer_image(self,img):
		img,(org_h,org_w) = self.load_image(img)
		img = paddle.to_tensor(img).unsqueeze(0)
		return self.infer(img,shape_list=[(org_h,org_w)])

	def visualize(self,images,polys):
		
		for img,poly in zip(images,polys):
			poly = poly['points']
			visualize(img, poly)
if __name__ == '__main__':
	config_path = "template.yaml"
	model = Model(config_path)
	
	x = paddle.randn((1,3,224,224))
	print(model.infer(x))
	
	image_path = r"images/test_data.jpg"
	visualize_img = cv2.imread(image_path)
 
	polys = model.infer_image(image_path)
	print(polys)
 
	visualize_img = cv2.imread(image_path)
	model.visualize([visualize_img],polys)
	cv2.imwrite("visualize.jpg", visualize_img)
	
	