from flask import Flask, jsonify, redirect, request,Response, render_template, url_for, send_from_directory
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf, sys
import tensorflow.compat.v1 as tf
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import hashlib
import math

app = Flask(__name__)
CORS(app)
uploadpath = "uploads/"

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webupload',methods=['POST'])
def webupload():
	file = request.files['file']
	filename = secure_filename(file.filename)
	file.save(os.path.join(uploadpath, filename))
	
	final = os.path.join(uploadpath, filename)
	Newlable = calculate(final)
	return render_template("result.html", results = Newlable)


def calculate(final):
	output_labels =[]
	image_data = tf.gfile.FastGFile(final, 'rb').read()
	label_lines = [line.rstrip() for line in tf.gfile.GFile("./Tensorflow/labels.txt")]
	with tf.gfile.FastGFile("./Tensorflow/Graph.pb", 'rb') as graphread:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(graphread.read())
		_ = tf.import_graph_def(graph_def, name='')
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')			
		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})			
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]			
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			print('%s (score = %.5f)' % (human_string, score))
			output_labels.append((human_string, str(round((score* 100),2))))
	print(output_labels)
	return output_labels

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=int(9999),debug=True, threaded=True)
