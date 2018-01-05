import numpy as np
import cv2
import os
import tensorflow as tf
import sys
import math
import pickle
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def normalise(x):
	mean = np.mean(x)
	std = np.std(x)
	std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
	y = np.multiply(np.subtract(x, mean), 1/std_adj)
	return y  

def load_model(model):
	model_exp = os.path.expanduser(model)
	if (os.path.isfile(model_exp)):
		print('Model filename: %s' % model_exp)
		with gfile.FastGFile(model_exp,'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')

def classify(img):
	with tf.Graph().as_default():

		with tf.Session() as sess:
			
			np.random.seed(42)
			
			# Load the model
			print('Loading feature extraction model')
			load_model('models/20170512-110547.pb')
			
			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]
			
			# Run forward pass to calculate embeddings
			print('Calculating features for images')
			emb_array = np.zeros((1, embedding_size))
			images = np.zeros((1, 160, 160, 3))
			img = normalise(img)
			images[0,:,:,:] = img
			feed_dict = { images_placeholder:images, phase_train_placeholder:False }
			emb_array = sess.run(embeddings, feed_dict=feed_dict)
			
			classifier_filename_exp = 'models/my_classifier.pkl'

			# Classify images
			print('Testing classifier')
			with open(classifier_filename_exp, 'rb') as infile:
				(model, class_names) = pickle.load(infile)

			print('Loaded classifier model from file "%s"' % classifier_filename_exp)

			predictions = model.predict_proba(emb_array)
			print(predictions)
			best_class_indices = np.argmax(predictions, axis=1)
			best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
			
			for i in range(len(best_class_indices)):
				print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
			return predictions, class_names				


def predict(test_image):
	face_present = False
	modi_present = False
	kejriwal_present = False

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	img = cv2.imread(test_image)
	if img is None:
		return img, face_present, kejriwal_present, modi_present
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces_dec = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces_dec:
		face_present = True
		face = img[y:y+h, x:x+w]
		# if face.shape[0] < 160:
		face = cv2.resize(face, (160,160))
		pred, cl = classify(face)
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		print(np.argmax(pred, axis=1))
		if max(pred[0]) >= 0.7:
			clno = np.argmax(pred, axis=1)[0]
			if clno == 0:
				kejriwal_present = True
			elif clno == 1:
				modi_present = True
			text = cl[np.argmax(pred, axis=1)[0]]
			cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	return img, face_present, kejriwal_present, modi_present

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('upload_done', filename=filename))
	return render_template('home.html')

@app.route('/<filename>', methods=['GET'])
def upload_done(filename):
	test_image = "uploads/" + filename
	predicted_image, face_p, kejriwal_p, modi_p = predict(test_image)
	if predicted_image is not None:
		cv2.imwrite("static/outputs/"+filename, predicted_image)
	# return send_from_directory(app.config['UPLOAD_FOLDER'],
	#                            filename)
	print("static/outputs/"+filename)
	return render_template('display_result.html', dis_img="static/outputs/"+filename, face_p=face_p, kejriwal_p=kejriwal_p, modi_p=modi_p)

if __name__ == '__main__':
	app.run(debug=True)