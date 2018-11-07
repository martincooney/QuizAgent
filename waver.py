#!/usr/bin/env python
import numpy as np
import cv2
from subprocess import call
import sys
import re
from datetime import date, datetime, timedelta

class waver_interface:
	def __init__(self):

		self.interactionState= 0

		self.windowHeight=1080 
		self.windowWidth= 1920 
		self.x_offset=0
		self.y_offset=0
		self.x_offset_quiz=960
		self.face_width=self.x_offset_quiz
		self.quiz_button_width=self.x_offset_quiz/2
		self.quiz_pane_width=self.x_offset_quiz

		self.startScreen = np.zeros((self.windowHeight, self.windowWidth, 3), dtype=np.uint8)
		self.contourSum= 0
		self.firstFrame = None 
		self.lastThresh = None

		(self.openCVVersion, _, _) = cv2.__version__.split(".")


def main():

	my_wav = waver_interface()

	cv2.namedWindow("waver_screen", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("waver_screen", cv2.WND_PROP_FULLSCREEN, 1) 

	my_wav.current_screen_image= my_wav.startScreen
	cv2.waitKey(100)

	camera = cv2.VideoCapture(0)

	totalYes = 0
	totalNo = 0
	max = 999999 #for my camera this is not ever reached, but you can control this as you like...

	(grabbed, frame) = camera.read()
	# check to see if we have reached the end of the video or if there is some problem
	if not grabbed:
		return -1

	(grabbed, frame) = camera.read() #first frame from camera tends to be not good, so get a few times...
	cv2.waitKey(10)		
	(grabbed, frame) = camera.read() 
	cv2.waitKey(10)
	(grabbed, frame) = camera.read() 
	cv2.waitKey(10)

	while True:

		(grabbed, frame) = camera.read()

		# check to see if we have reached the end of the video or if there is some problem
		if not grabbed:
			break

		resized_frame = cv2.resize(frame, (my_wav.face_width, 600))
		gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21,21), 0)

		if my_wav.firstFrame is None:
			print "Initializing background"
			my_wav.firstFrame = gray

		else:

			frameDelta=cv2.absdiff(my_wav.firstFrame, gray)
			thresh= cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	
			thresh= cv2.dilate(thresh, None, iterations=2)
	
			if my_wav.lastThresh is None:
				my_wav.lastThresh = thresh

			else:
				res = cv2.bitwise_or(my_wav.lastThresh, thresh)
				my_wav.lastThresh=res

				if(my_wav.openCVVersion=="2"):
					(cnts, _) = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				
				else:
					(_, cnts, _) = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				
				my_wav.contourSum=0
				for c in cnts:
					my_wav.contourSum+= cv2.contourArea(c)
					if(my_wav.contourSum> max):
						my_wav.contourSum= max

				res = cv2.merge((res,res,res))
				my_wav.startScreen[my_wav.y_offset:my_wav.y_offset+600, 0:my_wav.face_width] = res
				cv2.imshow("waver_screen", my_wav.current_screen_image)

				cv2.rectangle(my_wav.startScreen, (0, 900), (1800, 1100), (0, 0, 0), thickness=-1)
				cv2.putText(my_wav.startScreen, "i: init, 1: yes, 2: no, o: output, n: new, q: quit", (200, 1050), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 255), 5)


		originalKeyPress = cv2.waitKey(10)
		key= originalKeyPress & 0xFF

		if(key== ord("q")): 
			break
		elif(key== ord("i")):
			print "Re-initialize: forget previous motions"
			my_wav.firstFrame = gray
			my_wav.lastThresh = None
		elif(key== ord("1")):
			print "Gather score for yes"
			totalYes=my_wav.contourSum
			cv2.rectangle(my_wav.startScreen, (1000, 0), (1800, 400), (0, 0, 0), thickness=-1)
			cv2.putText(my_wav.startScreen, str(totalYes), (1000, 210), cv2.FONT_HERSHEY_SIMPLEX, 5, (100, 255, 100), 5)
		elif(key== ord("2")):
			print "Gather score for no"
			totalNo=my_wav.contourSum
			cv2.rectangle(my_wav.startScreen, (1000, 400), (1800, 700), (0, 0, 0), thickness=-1)
			cv2.putText(my_wav.startScreen, str(totalNo), (1000, 510), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 100, 100), 5)
		elif(key== ord("o")):
			cv2.rectangle(my_wav.startScreen, (1000, 700), (1800, 1200), (0, 0, 0), thickness=-1)
			print "Total Motion for Yes: ", totalYes
			print "Toal Motion for No: ", totalNo
			if(totalYes>totalNo):
				print "Decision: yes"
				cv2.putText(my_wav.startScreen, "yes", (1000, 810), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
			else:
				print "Decision: no"
				cv2.putText(my_wav.startScreen, "no", (1000, 810), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
		elif(key== ord("n")):
			print "Renew: clear all previous results"
			my_wav.firstFrame = gray
			my_wav.lastThresh = None
			totalYes=0
			totalNo=0
			cv2.rectangle(my_wav.startScreen, (1000, 0), (1800, 400), (0, 0, 0), thickness=-1)
			cv2.rectangle(my_wav.startScreen, (1000, 400), (1800, 700), (0, 0, 0), thickness=-1)
			cv2.rectangle(my_wav.startScreen, (1000, 700), (1800, 1200), (0, 0, 0), thickness=-1)

	cv2.destroyAllWindows()
	print "Closing program"

if __name__== '__main__':

    	print '-------------------------------------'
    	print '-            WAVER                  -'
    	print '-   SEP 2018, HH, Martin Cooney     -'
    	print '-------------------------------------'
	print "i: init, 1: yes, 2: no, o: output, n: new, q: quit"

	main()




