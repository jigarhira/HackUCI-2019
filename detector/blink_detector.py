import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import picamera
import picamera.array
from PIL import Image
from gpiozero import LED
from flask_socketio import SocketIO, send, emit
from flask_script import Manager, Server
from flask import Flask, jsonify, send_from_directory, render_template
from multiprocessing import Process, Array
from collections import defaultdict
from twilio.rest import Client
import subprocess
from subprocess import call
from threading import Thread
import os
import random

# eye detection delays
sleeping_delay = 100
sms_delay = 500


# LED pins
blue = LED(18)
green = LED(14)

# twilio account information (fill this in)
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)
sms_from = ''
sms_to = ''

KEY_INDICES = {
	"are_eyes_open": 0,
	"blink_count": 1
}

class Camera:
	def __init__(self, predictor, cascade, socket):
		self.predictor = predictor
		self.cascade = cascade
		self.asleep = False
		self.socket = socket
		self.is_running = False

	def play_music(self):
		rando = random.randint(1,2)
		os.system('aplay -D bluealsa:HCI=hci0,DEV=2C:41:A1:AD:A9:D9,PROFILE=a2dp /home/pi/Downloads/%d.wav'%rando)
		

	def start(self, arr):
		self.is_running = True
		# turns on the green power led
		green.on()
		# open the camera,load the cnn model
		model = load_model('blinkModel.hdf5')

		sent = 0; # if SMS has been sent

		# blinks is the number of total blinks ,close is
		# the counter for consecutive close predictions
		# and mem_counter the counter of the previous loop
		close = 0
		#links = 0
		mem_counter = 0
		state = ''
		time_elapsed = 0
		start = time.time()
		camera = picamera.PiCamera()

		output = picamera.array.PiRGBArray(camera)
		camera.framerate = 80
		camera.resolution = (320, 180)

		while self.is_running:
			now = time.time()

			camera.capture(output, 'rgb')
			frame = output.array
			output.truncate(0)
			frame = cv2.resize(frame, (1280, 720))
			eyes = self.crop_eyes(frame)

			# show the frame
			cv2.imshow('blinks counter', frame)
			key = cv2.waitKey(1) & 0xFF

			if eyes is None:
				print ("no eyes")
				continue
			else:
				left_eye,right_eye = eyes

			# average the predictions of the two eyes
			prediction = (model.predict(self.cnn_preprocess(left_eye)) + model.predict(self.cnn_preprocess(right_eye)))/2.0

			# blinks
			# if the eyes are open reset the counter for close eyes
			print("prediction: " + str(prediction))
			if prediction > 0.5:
				print("ya boi")
				# Emit when driver re-opens eyes
				if close == 1:
					self.socket.emit("openEyes", {'data': 1337})
				arr[KEY_INDICES["are_eyes_open"]] = True
				close = 0
				time_elapsed = 0
				start = time.time()
				sent = 0
				blue.off() # turn off the blue led
			else:
				arr[KEY_INDICES["are_eyes_open"]] = False
				time_elapsed += time.time() - start
				print(time_elapsed)
				self.socket.emit("closeEyes", {'data': 1337})
				if (time_elapsed > sleeping_delay) and (close == 0):
					#play like rick rolld or some shit
					#p = Process(target=self.play_music)
					#p.start()
					print("before soundthread")
					soundThread = Thread(target=self.play_music)
					soundThread.start()
					print("after soundthread")

					self.asleep = True
					print("You're sleeping")
					blue.on()	# turns on the blue led
					close += 1
					self.socket.emit("sleep", {'data': 1337})
				if (time_elapsed > sms_delay) and (sent == 0):
					print("SMS Sent!")
					message = client.messages.create(
						body="Sensor detected eyes closed for longer than 10 seconds, come pick me up.",
						from_=sms_from,
						to=sms_to
					)
					sent = 1
					self.socket.emit("smsSent", {'data': 1337})



			if arr[KEY_INDICES["are_eyes_open"]] and mem_counter > 1:
				print ("got here")
			arr[KEY_INDICES["blink_count"]] += 1
			# keep the counter for the next loop
			mem_counter = close

			# draw the total number of blinks on the frame along with
			# the state for the frame
			cv2.putText(frame, "Blinks: {}".format(arr[0]), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "State: {}".format(state), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	def stop(self):
		self.is_running = False

	# detect the face rectangle
	def detect(self, img, minimumFeatureSize=(20, 20)):
		if self.cascade.empty():
			raise (Exception("There was a problem loading your Haar Cascade xml file."))
		rects = self.cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

		# if it doesn't return rectangle return array
		# with zero lenght
		if len(rects) == 0:
			return []

		#  convert last coord from (width,height) to (maxX, maxY)
		rects[:, 2:] += rects[:, :2]

		return rects

	def crop_eyes(self, frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect the face at grayscale image
		te = self.detect(gray, minimumFeatureSize=(80, 80))

		# if the face detector doesn't detect face
		# return None, else if detects more than one faces
		# keep the bigger and if it is only one keep one dim
		if len(te) == 0:
			return None
		elif len(te) > 1:
			face = te[0]
		elif len(te) == 1:
			[face] = te

		# keep the face region from the whole frame
		face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
									right = int(face[2]), bottom = int(face[3]))

		# determine the facial landmarks for the face region
		shape = self.predictor(gray, face_rect)
		shape = face_utils.shape_to_np(shape)

		#  grab the indexes of the facial landmarks for the left and
		#  right eye, respectively
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# extract the left and right eye coordinates
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		# keep the upper and the lower limit of the eye
		# and compute the height
		l_uppery = min(leftEye[1:3,1])
		l_lowy = max(leftEye[4:,1])
		l_dify = abs(l_uppery - l_lowy)

		# compute the width of the eye
		lw = (leftEye[3][0] - leftEye[0][0])

		# we want the image for the cnn to be (26,34)
		# so we add the half of the difference at x and y
		# axis from the width at height respectively left-right
		# and up-down
		minxl = (leftEye[0][0] - ((34-lw)/2))
		maxxl = (leftEye[3][0] + ((34-lw)/2))
		minyl = (l_uppery - ((26-l_dify)/2))
		maxyl = (l_lowy + ((26-l_dify)/2))

		# crop the eye rectangle from the frame
		left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
		left_eye_rect = left_eye_rect.astype(int)
		left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

		# same as left eye at right eye
		r_uppery = min(rightEye[1:3,1])
		r_lowy = max(rightEye[4:,1])
		r_dify = abs(r_uppery - r_lowy)
		rw = (rightEye[3][0] - rightEye[0][0])
		minxr = (rightEye[0][0]-((34-rw)/2))
		maxxr = (rightEye[3][0] + ((34-rw)/2))
		minyr = (r_uppery - ((26-r_dify)/2))
		maxyr = (r_lowy + ((26-r_dify)/2))
		right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
		right_eye_rect = right_eye_rect.astype(int)
		right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

		# if it doesn't detect left or right eye return None
		if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
			return None
		# resize for the conv net
		left_eye_image = cv2.resize(left_eye_image, (34, 26))
		right_eye_image = cv2.resize(right_eye_image, (34, 26))
		right_eye_image = cv2.flip(right_eye_image, 1)
		# return left and right eye
		return left_eye_image, right_eye_image

	# make the image to have the same format as at training
	def cnn_preprocess(self, img):
		img = img.astype('float32')
		img /= 255
		img = np.expand_dims(img, axis=2)
		img = np.expand_dims(img, axis=0)
		return img


# Flask stuff
app = Flask(__name__, template_folder='../site/', static_url_path='', static_folder='../site/')
app.debug = True
print("App started")
socketio = SocketIO(app)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
c = Camera(predictor, face_cascade, socketio)
shared_data = Array('i',(0,0))

@app.route('/')
def index():
	global c, shared_data
	p = Process(target=c.start, args=(shared_data,))
	p.start()
	print ("???")
	return render_template('index.html')
	#return app.send_static_file('../site/index.html')

@app.route('/stats')
def stats():
	global shared_data
	return jsonify(blink_count=shared_data[KEY_INDICES["blink_count"]], are_eyes_open=shared_data[KEY_INDICES["are_eyes_open"]])

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file('../site/')

@socketio.on('data')
def on_data(a):
	global shared_data
	data = { "blink_count": shared_data[KEY_INDICES["blink_count"]], "are_eyes_open": shared_data[KEY_INDICES["are_eyes_open"]] }
	emit(data)

@socketio.on('phoneNumberUpdate')
def on_phone_number_update(phone):
	global sms_to
	sms_to = phone

def main():

#	print("Starting yo juj")

	#cameraThread = CameraThread(predictor, cascade)
	#cameraThread.start()
	print("Camera thread started")
	socketio.run(app)
	print("WS server started")
	socketio.emit('some event', {'data': 42})
	# do a little clean up
	#cv2.destroyAllWindows()
	#del(camera)

if __name__ == "__main__":
	main()
