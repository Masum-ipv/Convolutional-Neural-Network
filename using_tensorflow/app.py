from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
import numpy as np
import re, io, base64, cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils import predict
import json

# Default output
res = {"result": 0,
       "data": [], 
	   "error": ''}

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def main():
	if request.method == 'GET':
		return render_template('index.html')

	if request.method == 'POST':
		parameters = joblib.load('parameters_test.pkl')
		data=request.stream.read().decode('utf-8')

		# Convert data url to numpy array
		imgdata = data.split(',')[1]
		#print("imgdata ", imgdata)
		image_bytes = io.BytesIO(base64.b64decode(imgdata))
		im = Image.open(image_bytes)
		my_image = np.array(im)[:,:,0:3]
		#my_image = (255 - my_image)
		my_image = cv2.subtract(255, my_image)
		#plt.imshow(my_image)
		#plt.show()
		# Normalize and invert pixel values
		my_image = my_image.reshape((3072, 1))
		my_image = my_image / 255.
		#print(my_image.shape)
		my_image_prediction = predict(my_image, parameters)
		#print("my_image_prediction ", my_image_prediction)

		# Return label data
		res['result'] = 1
		res['data'] = [float(num) for num in my_image_prediction] 
		print(json.dumps(res))
		return json.dumps(res)


if __name__ == "__main__":
	app.run(debug=True)
