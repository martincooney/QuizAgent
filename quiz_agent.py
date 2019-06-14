#!/usr/bin/env python

'''
Copyright 2019 Martin Cooney (Work done at HPC/ITE, Halmstad University)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import rospy
import numpy as np
import cv2
from std_msgs.msg import String
import re
from datetime import date, datetime, timedelta
import time
from sound_play.libsoundplay import SoundClient
import os.path



class virtual_agent_interface:
	def __init__(self):

		#ROS 
		self.r=rospy.Rate(10)
		self.r.sleep()
		self.subSpeech = rospy.Subscriber('/recognizer/output', String, self.talkback)
    		self.soundhandle = SoundClient()


		#agent state
		self.speechRecognitionFlag = True 
		self.interactionState= 0
		self.asleepState = False
		self.codeSnippet = ""
		self.answerSnippet = ""
		self.execFlag = False
		self.lastTimeUserDidSomething = datetime.now()
		self.lastTimeAgentDidSomething = self.lastTimeUserDidSomething
		self.lastGazeVec_X = 0
		self.lastGazeVec_Y = 0
		self.agentEmotion = "happy"
		self.SHOULD_GRAB_A_PHOTO = False
		self.agentValence=0.5
		self.eyeClickedUtterances = ["Hey, that's my eyes!", "Ouch, stop clicking my eyes", "My eyes are very sensitive you know", "Help, help!", "The paiiiin, the paiiiin"]
		self.eyeClickedUtteranceLast= 0
		self.faceClickedUtterances = ["Stop slapping me!", "Didn't your mother teach you that it's not nice to touch people's faces", 
			"Are you looking for something to do?", "Oh, that actually felt kind of good"]
		self.faceClickedUtteranceLast= 0
		self.proactiveUtterancesAfterQuiz = ["Good question, eh?", "Do you see why that is the answer?", "Nice weather, huh?", 
			"Are you waiting for something more?", "How about trying another quiz?"]	
		self.proactiveUtterancesAfterQuizLast= 0
		self.teacherFace = cv2.imread("/home/turtlebot/ros_ws/src/hpc/src/agent_data/martin1.png") 
		self.teacherFace= cv2.resize(self.teacherFace, (360, 400))
		self.face_cascade = cv2.CascadeClassifier('/home/turtlebot/ros_ws/src/hpc/src/agent_data/haarcascade_frontalface_alt_tree.xml') 
		self.numberOfSecondsUntilAutoLogout = 30     #time values are set much too short, just for the purpose of demonstration (can be 180 or more)
		self.numberOfSecondsUntilHint= 5             #60
		self.numberOfSecondsUntilProactiveSpeech= 5  #10

		#agent visual parameters
		self.skinColor = (255, 200, 200)
		self.eyeColor = (100, 0, 0)
		self.eyebrowColor = (200, 100, 100)
		self.mouthColor = (200, 100, 100)

		self.baseFaceX1 = 300
		self.baseFaceX2 = 660
		self.baseFaceY1 = 100 
		self.baseFaceY2 = 500 
		self.baseEye_LeftX = 380  #"left" here means close to the left side of the canvas (agent's right eye)
		self.baseEye_RightX = 580
		self.baseEye_Y = 250
		self.baseEyebrow_LeftX1=330
		self.baseEyebrow_LeftX2=430
		self.baseEyebrow_RightX1=530
		self.baseEyebrow_RightX2=630
		self.baseEyebrow_Y= 150
		self.baseEyebrow_YHigh= 100
		self.baseMouth_X1= 330
		self.baseMouth_X2= 480
		self.baseMouth_X3= 630
		self.baseMouth_Y= 450
		self.baseMouth_YHigh= 400


		#interface width and height parameters
		self.windowHeight=1080 
		self.windowWidth= 1920
		self.x_offset=0
		self.y_offset=0
		self.x_offset_quiz=960
		self.face_width=self.x_offset_quiz
		self.quiz_button_width=self.x_offset_quiz/2
		self.quiz_pane_width=self.x_offset_quiz
		self.quizYSpace=50
		self.quizButtonHeight=100
		self.largeButtonCoordinates = [200, 600, 800, 800]
		self.button1Coordinates = [0, 600, 512, 800]
		self.button2Coordinates = [512, 600, 1024, 800]
		self.quizButtonCoordinates = [	0, 0+self.quizYSpace, 512, 0+self.quizYSpace, 
						0, self.quizButtonHeight+self.quizYSpace, 512, self.quizButtonHeight+self.quizYSpace,
						0, 2*self.quizButtonHeight+self.quizYSpace, 512, 2*self.quizButtonHeight+self.quizYSpace,
						0, 3*self.quizButtonHeight+self.quizYSpace, 512, 3*self.quizButtonHeight+self.quizYSpace,
						0, 4*self.quizButtonHeight+self.quizYSpace, 512, 4*self.quizButtonHeight+self.quizYSpace]
		self.startScreen = np.zeros((self.windowHeight, self.windowWidth, 3), dtype=np.uint8)
		self.quizAvailableScreen = np.zeros((self.windowHeight, self.windowWidth, 3), dtype=np.uint8)
		self.quizScreen = np.zeros((self.windowHeight, self.windowWidth, 3), dtype=np.uint8)
		self.quizAnswersScreen = np.zeros((self.windowHeight, self.windowWidth, 3), dtype=np.uint8)
		self.codeQuizScreen = np.zeros((self.windowHeight, self.windowWidth, 3), dtype=np.uint8)
		self.faceCanvas = np.zeros((600, 960, 3), dtype=np.uint8)
		self.hhLogo = cv2.imread("/home/turtlebot/ros_ws/src/hpc/src/hh3.jpg") 
		self.hhLogo= cv2.resize(self.hhLogo, (100, 100))


		#user and quiz parameters
		self.userName = ""
		self.userFirstName="default"
		self.listOfNames=[]
		self.quizStartDates=[] 
		self.quizEndDates=[] 
		self.quizAnswers=[] 
		self.quizHints=[] 
		self.courseNames=[] 
		self.quizScreens_questions = []
		self.quizScreens_answers = []
		self.selectedQuiz=-1
		self.selectedQuizName=""
		self.studentIDs=[]
		self.studentNames=[]
		self.studentCourses=[]
		self.namesOfCoursesWithQuizzes=[]
		self.quizNamesPerCourse=[]
		self.quizStartDatesPerCourse=[]
		self.quizEndDatesPerCourse=[]
		self.quizAnswersPerCourse=[]
		self.quizHintsPerCourse=[]
		self.successRate = 0.0
		self.correctAnswers = 0.0 
		self.numberOfAnsweredQuestions = 0.0
		self.startedQuiz=-1
		self.endedQuiz=-1
		self.answerCorrectness=-1
		self.hintGiven = False


		#initial processing: 
		#draw
		cv2.putText(self.startScreen, "Start", (50, 730), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
		cv2.rectangle(self.startScreen, (0, 600), (512, 800), (255, 255, 255), thickness=2)
		cv2.putText(self.startScreen, "Clear", (600, 730), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
		cv2.rectangle(self.startScreen, (512, 600), (1024, 800), (255, 255, 255), thickness=2)
		cv2.putText(self.startScreen, "Type the last 4 numbers of your student number then click start", (50, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	
		cv2.rectangle(self.quizAvailableScreen, (0, 600), (512, 800), (255, 255, 255), thickness=2)
		cv2.putText(self.quizAvailableScreen, "Back", (600, 730), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
		cv2.rectangle(self.quizAvailableScreen, (512, 600), (1024, 800), (255, 255, 255), thickness=2)
		cv2.putText(self.quizAvailableScreen, "Click on a quiz above to see it", (50, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


		cv2.putText(self.quizScreen, "Answers", (50, 730), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
		cv2.rectangle(self.quizScreen, (0, 600), (512, 800), (255, 255, 255), thickness=2)
		cv2.putText(self.quizScreen, "Back", (600, 730), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
		cv2.rectangle(self.quizScreen, (512, 600), (1024, 800), (255, 255, 255), thickness=2)


		cv2.putText(self.quizAnswersScreen, "Back", (600, 730), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
		cv2.rectangle(self.quizAnswersScreen, (512, 600), (1024, 800), (255, 255, 255), thickness=2)

		self.startScreen[0:100, 0:100] = self.hhLogo

		#populate 
		self.parseUserInformation() 
		self.parseQuizInformation()


	def moveAWeeBit(self, xAmount, yAmount):

		#check that the motion is possible
		if (	(self.baseFaceX1 + xAmount) > 0 and (self.baseFaceX2 + xAmount) < 700 and
			(self.baseFaceY1 + yAmount) > 0 and (self.baseFaceY2 + xAmount) < 600
		):

			self.baseFaceX1 += xAmount
			self.baseFaceX2 += xAmount
			self.baseFaceY1 += yAmount
			self.baseFaceY2 += yAmount

			self.baseEye_LeftX += xAmount 
			self.baseEye_RightX += xAmount
			self.baseEye_Y += yAmount

			self.baseEyebrow_LeftX1+= xAmount
			self.baseEyebrow_LeftX2+= xAmount
			self.baseEyebrow_RightX1+= xAmount
			self.baseEyebrow_RightX2+= xAmount
			self.baseEyebrow_Y+= yAmount
			self.baseEyebrow_YHigh+= yAmount

			self.baseMouth_X1+= xAmount
			self.baseMouth_X2+= xAmount
			self.baseMouth_X3+= xAmount
			self.baseMouth_Y+= yAmount
			self.baseMouth_YHigh+= yAmount


	def showQuizImage(self):
		self.r.sleep()
		if(len(self.listOfNames) > self.selectedQuiz):
			myImage= self.quizScreens_questions[self.selectedQuiz]
			resizedImage= cv2.resize(myImage, (960, 600))
			self.quizScreen[self.y_offset:self.y_offset +600, self.x_offset_quiz:self.x_offset_quiz+960] = resizedImage 

			cv2.putText(self.quizScreen, "Type your answer and press return", (50, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

			self.interactionState = 2	
			self.selectedQuizName = self.listOfNames[self.selectedQuiz]

		else:
			print "This quiz is not available."


	def talkback(self, data):
		#print data.data
		self.r.sleep()
		if(self.speechRecognitionFlag == False):
			print "heard something but ignoring"
		else:
			print "heard", data.data
			if(data.data=='hello'):
				print "Hello to you too"
				self.soundhandle.say('Hello to you too!')
			elif(data.data=='goodbye'):
				print "Goodbye"
				self.soundhandle.say('Goodbye!')
			elif(data.data=='bye bye'):
				print "Bye bye"
				self.soundhandle.say('Bye bye!')
			elif(data.data=='thank you'):
				print ""
				self.soundhandle.say("You're welcome!")
			elif(data.data=='how are you'):
				print ""
				self.soundhandle.say("I'm fine, thank you. How are you?")
			elif(data.data=='handsome' or data.data=='cool'):
				print ""
				self.soundhandle.say("Thank you!")


	def showImageBasedOnState(self):

		if(self.interactionState==0):
			self.current_screen_image= self.startScreen 
		elif(self.interactionState==1):
			self.current_screen_image= self.quizAvailableScreen
		elif(self.interactionState==2):
			self.current_screen_image= self.quizScreen
		elif(self.interactionState==3):
			self.current_screen_image= self.quizAnswersScreen


	def getCurrentScreenImageBasedOnState(self):

		if(self.interactionState==0):
			return self.startScreen
		elif(self.interactionState==1):
			return self.quizAvailableScreen
		elif(self.interactionState==2):
			return self.quizScreen
		elif(self.interactionState==3):
			return self.quizAnswersScreen
		return self.startScreen


	def parseUserInformation(self):
		#format: ID. Name. {course1, course2, ...}.
		#parse file, set list of courses...

		f=open('/home/turtlebot/ros_ws/src/hpc/src/agent_data/agent_personData.txt', 'r')
		l=list(f)
		f.close()

		if len(l) > 0:
			for currentLine in l:

				currentLine= currentLine.rstrip()
				currentLineAsList = re.split(r'[.]', currentLine)
				studentID = currentLineAsList[0]
				studentName = currentLineAsList[1].lstrip()
				courses = currentLineAsList[2].lstrip()
				courses = courses[1:-1] #remove braces
				theCourses = re.split(r'[,]', courses)

				self.studentIDs.append(studentID)
				self.studentNames.append(studentName)
				self.studentCourses.append(theCourses)


	def parseQuizInformation(self):
		#format: ID. Name. {course1, course2, ...}.
		#parse file, set list of courses...

		f=open('/home/turtlebot/ros_ws/src/hpc/src/agent_data/agent_quizData.txt', 'r')
		l=list(f)
		f.close()
		#print(len(l))
		if len(l) > 0:
			i=0
			while ( i < len(l)):

				#read name of course
				#make blank lists for quiz names, dates, etc
				#read number of quizzes
				#read in this number of lines
				#store everything

				currentLine= l[i].rstrip()
				#print "first Line ", currentLine
				currentLineAsList = re.split(r'[ ]', currentLine)
				#print currentLineAsList
				courseName = currentLineAsList[0]
				#print "courseName", courseName
				quizNames=[]
				quiz_StartDates=[]
				quiz_EndDates=[]
				quizAnswers=[]
				quizHints=[]


				numberOfQuizzes = currentLineAsList[1]
				#print "numberOfQuizzes", numberOfQuizzes
				i = i+1
				if int(numberOfQuizzes) > 0:
					for j in range(int(numberOfQuizzes)):

						currentLine= l[i+j].rstrip()
						#print "currentLine ", currentLine
						currentLineAsList = re.split(r'[,]', currentLine)
						quizNames.append(currentLineAsList[0])
						quiz_StartDates.append(currentLineAsList[1])
						quiz_EndDates.append(currentLineAsList[2])

						#process answers
						answersString = currentLineAsList[3][1:-1] #remove braces
						answersList = re.split(r'[;]', answersString)
						quizAnswers.append(answersList)

						hintString = currentLineAsList[4][1:-1] #remove quotes
						quizHints.append(hintString)

					i = i+ int(numberOfQuizzes)
				else:
					i = i+1

				self.namesOfCoursesWithQuizzes.append(courseName)
				self.quizNamesPerCourse.append(quizNames)
				self.quizStartDatesPerCourse.append(quiz_StartDates)
				self.quizEndDatesPerCourse.append(quiz_EndDates)
				self.quizAnswersPerCourse.append(quizAnswers)
				self.quizHintsPerCourse.append(quizHints)



	def parseUserHistorySimple(self):

		userHistoryFilename = '/home/turtlebot/ros_ws/src/hpc/src/agent_data/students/%s.txt' % self.userName
		fileExists = os.path.isfile(userHistoryFilename)
		if fileExists:
			f=open(userHistoryFilename, 'r')
			l=list(f)
			f.close()
			#print(len(l))

			self.successRate = 0.0
			self.correctAnswers = 0.0

			if len(l) > 0:
				for i in range(len(l)):

					currentLine= l[i].rstrip()
					currentLineAsList = re.split(r'[;]', currentLine)
					answerGrade = currentLineAsList[4].lstrip()
					if(answerGrade == "True"):
						self.correctAnswers +=1.0

 				self.numberOfAnsweredQuestions = float(len(l)) 
				self.successRate = self.correctAnswers / self.numberOfAnsweredQuestions
			else:
				self.numberOfAnsweredQuestions =  0.0

		else:
			print "file doesn't exist"
			f=open(userHistoryFilename, 'w') #let's create a blank file
			f.close()
			self.successRate = 0.0
			self.correctAnswers = 0.0
			self.numberOfAnsweredQuestions =  0.0


	def checkIfUserInDatabase(self, userId):

		#for loop, just check through user info if matching id...
		if len(self.studentIDs) > 0:
			for i, currentID in enumerate(self.studentIDs):
				if userId == currentID:
					self.userFirstName=self.studentNames[i]
					return True
		return False


	def getEmotionCategory(self): 
		if(self.agentValence> 0.67):
			return "happy"
		elif(self.agentValence> 0.33):
			return "neutral"
		return "sad"


	def processQuizButtonClick(self, x, y): 

		for i in range(10): #for simplicity we just allow 10 quizzes at a time right now
			if(x > (self.x_offset_quiz+ self.quizButtonCoordinates[i*2]) and x < (self.x_offset_quiz + self.quizButtonCoordinates[i*2]+512) 
				and y > self.quizButtonCoordinates[(i*2)+1] and y < (self.quizButtonCoordinates[(i*2)+1]+ self.quizButtonHeight)):

				self.selectedQuiz=i
				if(len(self.listOfNames) > self.selectedQuiz): #is this a valid quiz to click?

					print "User clicked quiz", i+1
					whatToSay = "quiz "+ str(i+1)
					self.soundhandle.say(whatToSay)
					self.startedQuiz = datetime.now()
					self.lastTimeAgentDidSomething=self.startedQuiz #this is not needed currently since hints are only given once, but will be helpful later
					#print self.startedQuiz
					self.showQuizImage()
					self.hintGiven = False
					

	def drawScores(self):

		currentScreen= self.getCurrentScreenImageBasedOnState()
		if((self.interactionState != 0) and (self.numberOfAnsweredQuestions != 0.0)):
			self.successRate = (self.correctAnswers / self.numberOfAnsweredQuestions) * 100.0
			textToWrite= "Score: %.2f %%" % self.successRate
			cv2.putText(currentScreen, textToWrite, (0, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


	def drawSelf(self):
		

		self.agentEmotion = self.getEmotionCategory()

		eyeWidth = 20
		eyebrowWidth = 20
		mouthWidth = 20

		BOUNCY= False 

		if(BOUNCY): #This is not currently used but can be modified to make the agent move a little bit over time to give an appearance of lifelikeness/agency
			randXMovement = randint(0, 20)
			self.baseFaceX1 += randXMovement
			self.baseFaceX2 += randXMovement

			randYMovement = randint(0, 20)
			self.baseFaceY1 += randYMovement
			self.baseFaceY2 += randYMovement

		cv2.rectangle(self.faceCanvas, (self.baseFaceX1, self.baseFaceY1), (self.baseFaceX2, self.baseFaceY2), (0, 0, 0), thickness=40) #clear outline for when face moves

		#draw face rectangle
		self.faceCanvas[self.baseFaceY1:self.baseFaceY2, self.baseFaceX1:self.baseFaceX2] = self.teacherFace

		cv2.rectangle(self.faceCanvas, (self.baseEye_LeftX-80, self.baseEye_Y-50), (self.baseEye_RightX+80, self.baseEye_Y+50), (100, 100, 100), thickness=-1) #make gray area for eyes, mouth
		cv2.rectangle(self.faceCanvas, (self.baseMouth_X1-30, self.baseMouth_Y-50), (self.baseMouth_X3+30, self.baseMouth_Y+50), (100, 100, 100), thickness=-1) 

		#draw eyes
		#can maybe just very simply check the time, if five seconds, blink
		currentSeconds= int((datetime.now() - datetime(1970,1,1)).total_seconds()) 

		if currentSeconds%5 == 0:
			cv2.line(self.faceCanvas,(self.baseEye_LeftX-self.lastGazeVec_X-20, self.baseEye_Y-self.lastGazeVec_Y),(self.baseEye_LeftX-self.lastGazeVec_X+20, self.baseEye_Y-self.lastGazeVec_Y),self.eyeColor,eyeWidth)
			cv2.line(self.faceCanvas,(self.baseEye_RightX-self.lastGazeVec_X-20, self.baseEye_Y-self.lastGazeVec_Y),(self.baseEye_RightX-self.lastGazeVec_X+20, self.baseEye_Y-self.lastGazeVec_Y),self.eyeColor,eyeWidth)
		else:
			cv2.circle(self.faceCanvas,(self.baseEye_LeftX-self.lastGazeVec_X, self.baseEye_Y-self.lastGazeVec_Y), 20, self.eyeColor, -1)
			cv2.circle(self.faceCanvas,(self.baseEye_RightX-self.lastGazeVec_X, self.baseEye_Y-self.lastGazeVec_Y), 20, self.eyeColor, -1)


		#draw eye brows
		if self.agentEmotion == "neutral":
			cv2.line(self.faceCanvas,(self.baseEyebrow_LeftX1,self.baseEyebrow_Y),(self.baseEyebrow_LeftX2,self.baseEyebrow_Y),self.eyebrowColor,eyebrowWidth) #eye brow left screen
			cv2.line(self.faceCanvas,(self.baseEyebrow_RightX1,self.baseEyebrow_Y),(self.baseEyebrow_RightX2,self.baseEyebrow_Y),self.eyebrowColor,eyebrowWidth) #eye brow right screen
		elif self.agentEmotion == "sad":
			cv2.line(self.faceCanvas,(self.baseEyebrow_LeftX1,self.baseEyebrow_Y),(self.baseEyebrow_LeftX2,self.baseEyebrow_YHigh),self.eyebrowColor,eyebrowWidth) 
			cv2.line(self.faceCanvas,(self.baseEyebrow_RightX1,self.baseEyebrow_YHigh),(self.baseEyebrow_RightX2,self.baseEyebrow_Y),self.eyebrowColor,eyebrowWidth) 
		elif self.agentEmotion == "angry":
			cv2.line(self.faceCanvas,(self.baseEyebrow_LeftX1,self.baseEyebrow_YHigh),(self.baseEyebrow_LeftX2,self.baseEyebrow_Y),self.eyebrowColor,eyebrowWidth) 
			cv2.line(self.faceCanvas,(self.baseEyebrow_RightX1,self.baseEyebrow_Y),(self.baseEyebrow_RightX2,self.baseEyebrow_YHigh),self.eyebrowColor,eyebrowWidth) 
		else:
			cv2.line(self.faceCanvas,(self.baseEyebrow_LeftX1,self.baseEyebrow_Y),(self.baseEyebrow_LeftX2,self.baseEyebrow_Y),self.eyebrowColor,eyebrowWidth) 
			cv2.line(self.faceCanvas,(self.baseEyebrow_RightX1,self.baseEyebrow_Y),(self.baseEyebrow_RightX2,self.baseEyebrow_Y),self.eyebrowColor,eyebrowWidth) 

		
		#draw mouth
		#mouth minimally requires 2 lines to encode happy, angry, neutral (3 dots, 1 triangle or line), but open mouth requires 4 (or circle); kiss, only 2 for now

		if self.agentEmotion == "neutral":
			cv2.line(self.faceCanvas,(self.baseMouth_X1,self.baseMouth_Y),(self.baseMouth_X2,self.baseMouth_Y),self.mouthColor,mouthWidth) #mouth line 1
			cv2.line(self.faceCanvas,(self.baseMouth_X2,self.baseMouth_Y),(self.baseMouth_X3,self.baseMouth_Y),self.mouthColor,mouthWidth) #mouth line 2
		elif self.agentEmotion == "happy":
			cv2.line(self.faceCanvas,(self.baseMouth_X1,self.baseMouth_YHigh),(self.baseMouth_X2,self.baseMouth_Y),self.mouthColor,mouthWidth) 
			cv2.line(self.faceCanvas,(self.baseMouth_X2,self.baseMouth_Y),(self.baseMouth_X3,self.baseMouth_YHigh),self.mouthColor,mouthWidth) 
		elif self.agentEmotion == "angry" or self.agentEmotion == "sad":
			cv2.line(self.faceCanvas,(self.baseMouth_X1,self.baseMouth_Y),(self.baseMouth_X2,self.baseMouth_YHigh),self.mouthColor,mouthWidth)
			cv2.line(self.faceCanvas,(self.baseMouth_X2,self.baseMouth_YHigh),(self.baseMouth_X3,self.baseMouth_Y),self.mouthColor,mouthWidth)
		else:
			cv2.line(self.faceCanvas,(self.baseMouth_X1,self.baseMouth_Y),(self.baseMouth_X2,self.baseMouth_Y),self.mouthColor,mouthWidth) 
			cv2.line(self.faceCanvas,(self.baseMouth_X2,self.baseMouth_Y),(self.baseMouth_X3,self.baseMouth_Y),self.mouthColor,mouthWidth) 


		#not face but using top left part of face canvas for this...
		self.faceCanvas[0:100, 10:110] = self.hhLogo
		cv2.putText(self.faceCanvas, "HPC QuizGiver", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1) 


		if (self.interactionState == 0):
			self.startScreen[self.y_offset:self.y_offset+600, 0:self.face_width] = self.faceCanvas
		elif (self.interactionState == 1):
			self.quizAvailableScreen[self.y_offset:self.y_offset+600, 0:self.face_width] = self.faceCanvas
		elif (self.interactionState == 2):
			self.quizScreen[self.y_offset:self.y_offset+600, 0:self.face_width] = self.faceCanvas
		elif(self.interactionState==3):
			self.quizAnswersScreen[self.y_offset:self.y_offset+600, 0:self.face_width] = self.faceCanvas



	def userClickedInterface(self, event, x, y, flags):

		NUMBER_OF_STATES= 4
		if event == cv2.EVENT_LBUTTONDOWN:
			#print "clicked"
			#print  x, y

			self.lastTimeUserDidSomething= datetime.now()

			if(x > 300 and x < 660 and y > 215 and y < 300):
				print "eye clicked"
				self.soundhandle.say(self.eyeClickedUtterances[self.eyeClickedUtteranceLast])
				self.eyeClickedUtteranceLast= (self.eyeClickedUtteranceLast + 1) % len(self.eyeClickedUtterances)


			elif(x > 300 and x < 660 and y > 100 and y < 500):
				print "face clicked"
				self.soundhandle.say(self.faceClickedUtterances[self.faceClickedUtteranceLast])
				self.faceClickedUtteranceLast= (self.faceClickedUtteranceLast + 1) % len(self.faceClickedUtterances)


			if (self.interactionState == 0):
				#print x, y

				if(x > self.button1Coordinates[0] and x < self.button1Coordinates[2] and y > self.button1Coordinates[1] and y < self.button1Coordinates[3] ):
					print "Logging in"

					#show quiz questions
					userName=self.userName
					weHaveAValidUser = self.checkIfUserInDatabase(userName) #this also sets user first name
					if weHaveAValidUser == False:
					
						print "Sorry, the agent can't recognize you, try again"
						cv2.rectangle(self.startScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
						self.userName = ""
						cv2.putText(self.startScreen, "Sorry, I can't recognize you, please try again", (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				

					else: 
						print "Recognized user", self.userFirstName
						self.interactionState = 1
						sayHelloString = "Hey " +  self.userFirstName + "!"
						self.soundhandle.say(sayHelloString)
				
						#clear 
						cv2.rectangle(self.quizAvailableScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
						cv2.rectangle(self.quizScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
						cv2.rectangle(self.quizAnswersScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
						#say hi
						cv2.putText(self.quizAvailableScreen, sayHelloString, (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
						cv2.putText(self.quizScreen, sayHelloString, (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
						cv2.putText(self.quizAnswersScreen, sayHelloString, (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

						#get list of courses for this user; for each course, get list of quizzes 
						print "Parsing quiz information for user"
						self.listOfNames=[] #clear lists
						self.quizStartDates=[]
						self.quizEndDates=[] 
						self.quizAnswers=[] 
						self.courseNames=[] 

						today=date.today()
						quizIndex=0

						for i in range(len(self.studentNames)):
							if(self.studentNames[i] == self.userFirstName):
								currentStudentsCourses =  self.studentCourses[i]
								#print "Found this user with courses:", currentStudentsCourses

						for i in range(len(currentStudentsCourses)):                    #for each course name the student is taking
							index = -1
							for j in range(len(self.namesOfCoursesWithQuizzes)):    #find this course in our list of all courses, get index
								if(currentStudentsCourses[i] == self.namesOfCoursesWithQuizzes[j]):
									index = j
							if(index!=-1):
								#print "quiz names:", self.quizNamesPerCourse[index] #uncomment for debugging
								#print "quiz start dates:", self.quizStartDatesPerCourse[index]
								for j in range(len(self.quizNamesPerCourse[index])): #for each quiz associated with each course
								
									quizStartDate= datetime.strptime(self.quizStartDatesPerCourse[index][j], '%Y-%m-%d')
									quizEndDate= datetime.strptime(self.quizEndDatesPerCourse[index][j], '%Y-%m-%d')
									#print "quiz date:", self.quizStartDatesPerCourse[index][j], quizStartDate.date().isoformat() 
									if (quizStartDate.date() < today) and (today < quizEndDate.date()):
										#print "today is greater than start date and less than end date"
										self.listOfNames.append(self.quizNamesPerCourse[index][j])
										self.quizStartDates.append(self.quizStartDatesPerCourse[index][j])	
										self.quizEndDates.append(self.quizEndDatesPerCourse[index][j])	
										self.quizAnswers.append(self.quizAnswersPerCourse[index][j])	
										self.quizHints.append(self.quizHintsPerCourse[index][j])	
										self.courseNames.append(self.namesOfCoursesWithQuizzes[index])
										quizIndex +=1
						#print self.listOfNames, self.quizStartDates, self.quizAnswers, self.quizHints, self.courseNames #uncomment for debugging

						self.quizScreens_questions = []
						self.quizScreens_answers = []
						for aName in self.listOfNames:
							fileName = "/home/turtlebot/ros_ws/src/hpc/src/agent_data/quizzes/" + aName + "_q.png"
							#print fileName		
							self.quizScreens_questions.append(cv2.imread(fileName))
							fileName = "/home/turtlebot/ros_ws/src/hpc/src/agent_data/quizzes/" + aName + "_a.png"
							#print fileName
							self.quizScreens_answers.append(cv2.imread(fileName))

						#display quizzes
						#for each quiz name, write it to screen in the appropriate place
						#for now keep it simple, only show a max of 10 quizzes
						print "Displaying quizzes"
						nameIndex = 0
						numberOfTimesToIterate = min(len(self.listOfNames), 10)

						for j in range(numberOfTimesToIterate):
							buttonText = self.courseNames[j] + " " + self.listOfNames[j]
							cv2.putText(self.quizAvailableScreen, buttonText, (487*(nameIndex%2) + 50+ self.x_offset_quiz, 50 + self.quizYSpace + (self.quizButtonHeight*int(nameIndex/2))), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255), 2) 
							cv2.rectangle(self.quizAvailableScreen, (self.x_offset_quiz + (((j+1)%2)*self.quiz_button_width), ((j/2)*self.quizButtonHeight) +self.quizYSpace), (self.x_offset_quiz+self.quiz_pane_width, (((j/2) + 1)*self.quizButtonHeight)+self.quizYSpace), (255, 255, 255), thickness=2)
							nameIndex+=1

						self.userName= userName

						#also read in this person's history
						self.parseUserHistorySimple()


				elif(x > (self.button2Coordinates[0]) and x < (self.button2Coordinates[2]) and y > self.button2Coordinates[1] and y < self.button2Coordinates[3] ):
					print "Clicked clear button"
					self.userName = ""
					cv2.rectangle(self.startScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
					self.interactionState = 0

			#if second state and user has pressed quiz, or cancel
			elif (self.interactionState == 1):
				#print "available quizzes"

				if(x > self.button2Coordinates[0] and x < self.button2Coordinates[2] and y > self.button2Coordinates[1] and y < self.button2Coordinates[3] ):
					print "Clicked back button"
					self.interactionState = 0
					#print self.interactionState
					self.userName = ""
					cv2.rectangle(self.startScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
					#here maybe need to erase more stuff
					cv2.rectangle(self.quizAvailableScreen, (800, 0), (2000, 600), (0, 0, 0), thickness=-1) #this not good need to just clear quizzes

				self.processQuizButtonClick(x, y)


			#if quiz state and user has pressed answers, or cancel
			elif (self.interactionState == 2):


				if(x > self.button1Coordinates[0] and x < self.button1Coordinates[2] and y > self.button1Coordinates[1] and y < self.button1Coordinates[3] ):
					print "Showing answer"
					self.soundhandle.say("here are the answers!")
					#check each item in answer list for selected quiz; if the entered string is correct, indicate the answer is correct, else false
					answerCorrect=False 

					for anAnswer in self.quizAnswers[self.selectedQuiz]:
						if anAnswer == self.answerSnippet:
							answerCorrect=True
					if answerCorrect==True:
						agentResponse= "Correct!"
						self.correctAnswers +=1.0 
						print "Correct"
					else:
						agentResponse= "Incorrect, good luck next time"
						print "Incorrect"

					self.numberOfAnsweredQuestions += 1.0

					cv2.rectangle(self.quizAnswersScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1) #clear
					cv2.putText(self.quizAnswersScreen, agentResponse, (30, 930), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

					self.endedQuiz= datetime.now()
					self.lastTimeAgentDidSomething= self.endedQuiz #refresh so agent doesn't immediately start talking
					timeTakenToAnswerQuestion = self.endedQuiz - self.startedQuiz
					duration_in_s = timeTakenToAnswerQuestion.total_seconds() 
					print "Time Taken To Answer Question %.2fs" % duration_in_s
					self.answerCorrectness=answerCorrect

					#open student file, add new entry for this quiz
					studentFileName = "agent_data/students/" + self.userName + ".txt"
				
					startTimeString = self.startedQuiz.strftime("%d/%m/%Y, %H:%M:%S")
					lineToWrite =  startTimeString + "; " +  self.courseNames[self.selectedQuiz] + "; " + self.listOfNames[self.selectedQuiz]  + "; " + self.answerSnippet + "; {};".format(answerCorrect) + " {}; ".format(duration_in_s) + str(self.quizAnswers[self.selectedQuiz]) + "\n"


					f=open(studentFileName, 'a')
					f.write(lineToWrite)
					f.close()

					self.answerSnippet=""
					#need to erase the old answer from the old screen as well
					cv2.rectangle(self.quizAvailableScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
					cv2.rectangle(self.quizScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)

					#show quiz answers
					myImage= self.quizScreens_answers[self.selectedQuiz]
					resizedImage= cv2.resize(myImage, (self.face_width, 600))
					self.quizAnswersScreen[self.y_offset:self.y_offset +600, self.x_offset_quiz:self.x_offset_quiz+self.face_width] = resizedImage 

					self.interactionState = 3

					today=date.today()

					#this stores all basic interactions by day (all users); details for each user are stored separately
					outputFileName= '/home/turtlebot/ros_ws/src/hpc/src/agent_data/logfiles/logfile-%s.txt' % (today.isoformat()) 
					keywordFile=open(outputFileName, 'a')
					lineToWrite= "%s %s %d %s %s %s\n" % (self.userName, self.userFirstName, self.selectedQuiz, self.selectedQuizName, today.isoformat(), datetime.now().time())
					keywordFile.write(lineToWrite)
					keywordFile.close()

					#self.SHOULD_GRAB_A_PHOTO = True #This can be uncommented if it is okay to take photos (if there is consent/etc)


				elif(x > self.button2Coordinates[0] and x < self.button2Coordinates[2] and y > self.button2Coordinates[1] and y < self.button2Coordinates[3] ):
					print "Clicked back button"
					self.interactionState = 1

			#if quiz answer state and user has pressed cancel
			elif (self.interactionState == 3):
				if(x > self.button2Coordinates[0] and x < self.button2Coordinates[2] and y > self.button2Coordinates[1] and y < self.button2Coordinates[3]  ):
					print "Clicked back button"
					self.interactionState = 1

			self.showImageBasedOnState() 


def userClickedInterfaceWrapper(event, x, y, flags, param):
	self= param
	self.userClickedInterface(event, x, y, flags)


def main():
	rospy.init_node('hpcagent', anonymous=True)

	my_agent = virtual_agent_interface()

	cv2.namedWindow("agent_screen", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("agent_screen", cv2.WND_PROP_FULLSCREEN, 1) 
	cv2.setMouseCallback("agent_screen", userClickedInterfaceWrapper, my_agent)

	vs = cv2.VideoCapture(0)
	time.sleep(2.0)

	my_agent.showImageBasedOnState()
	cv2.waitKey(100)

	WE_ARE_USING_A_CAMERA =True 
	SHOW_FACE_DETECTION=True 
	SHOW_TIREDNESS = False
	AUTO_LOGOUT = True
	lastKeyPress = -1


	while True:
		my_agent.r.sleep()
		cv2.imshow("agent_screen", my_agent.current_screen_image)

		#check if the computer has been inactive for a long time, then go back to start screen
		currentTime = datetime.now() 
		if(AUTO_LOGOUT and currentTime > (my_agent.lastTimeUserDidSomething + timedelta(seconds=my_agent.numberOfSecondsUntilAutoLogout))):
			print "Auto logout due to user inactivity"
			my_agent.userName = ""
			my_agent.answerSnippet=""
			cv2.rectangle(my_agent.startScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
			if(my_agent.interactionState != 0):
				cv2.putText(my_agent.startScreen, "Logged out of previous session automatically", (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				cv2.rectangle(my_agent.quizAvailableScreen, (800, 0), (2000, 600), (0, 0, 0), thickness=-1) 
				cv2.rectangle(my_agent.startScreen, (0, 550), (1024, 590), (0, 0, 0), thickness=-1)
				my_agent.numberOfAnsweredQuestions = 0.0 #clear the score as well
				my_agent.successRate = 0.0
				my_agent.correctAnswers = 0.0		
				my_agent.interactionState = 0
			print "auto logged out!", my_agent.interactionState
			my_agent.lastTimeUserDidSomething = currentTime
			my_agent.showImageBasedOnState()

		''' #this is not currently used, but can be adapted to make the agent fall asleep if no one is interacting with it
		if(SHOW_TIREDNESS and currentTime > (my_agent.lastTimeUserDidSomething + timedelta(seconds=5))):
			print "user inactive"
			#my_agent.startScreen[my_agent.y_offset:my_agent.y_offset+600, 0:my_agent.face_width] = my_agent.asleepFace #draw sleeping face
			my_agent.asleepState = True
			
		'''

		#if quiz question screen, and enough secs pass random chance for robot to say something
		if(my_agent.hintGiven == False and my_agent.interactionState == 2 and currentTime > (my_agent.lastTimeAgentDidSomething + timedelta(seconds=my_agent.numberOfSecondsUntilHint))):
			print "Agent gave hint"
			my_agent.soundhandle.say(my_agent.quizHints[my_agent.selectedQuiz])
			my_agent.lastTimeAgentDidSomething = currentTime 
			my_agent.hintGiven = True

		#if quiz answer already shown, also robot can say some stuff: "are you waiting for something more?" "do you see why this is the answer?" "good question eh?", etc, avoiding repetition	
		if(my_agent.interactionState == 3 and currentTime > (my_agent.lastTimeAgentDidSomething + timedelta(seconds=my_agent.numberOfSecondsUntilProactiveSpeech))):
			print "Agent proactively commented after quiz", my_agent.interactionState
			my_agent.soundhandle.say(my_agent.proactiveUtterancesAfterQuiz[my_agent.proactiveUtterancesAfterQuizLast])
			my_agent.proactiveUtterancesAfterQuizLast= (my_agent.proactiveUtterancesAfterQuizLast + 1) % len(my_agent.proactiveUtterancesAfterQuiz)
			my_agent.lastTimeAgentDidSomething = currentTime 

		if (WE_ARE_USING_A_CAMERA and my_agent.asleepState == False): 
			#check camera, process
			ret, frame = vs.read()
			if frame is None:
				break

			if(my_agent.SHOULD_GRAB_A_PHOTO):
				today=date.today()
				imageFileName = "/home/turtlebot/ros_ws/src/hpc/src/agent_data/photos/%s-%d-%s.jpg" % (my_agent.userName, my_agent.selectedQuiz, today.isoformat()) 
				cv2.imwrite(imageFileName, frame)
				my_agent.SHOULD_GRAB_A_PHOTO = False
			faceFound = 0
			frame = cv2.resize(frame, (my_agent.face_width, 600)) #960, 600
		
			frame = cv2.flip(frame, 1)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = my_agent.face_cascade.detectMultiScale(gray, 1.3, 5)

			for (x,y,w,h) in faces:

    				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) 
				cv2.line(frame,(x+(w/2),y+(h/2)),(480,300),(255,0, 0),5) #middle of 960, 600
				if faceFound == 0:
					faceFound = 1
					gazeVecX = ((480. - float(x+(w/2.)) )/ 480.)*70
					gazeVecY = ((300. - float(y+(h/2.)) )/ 300.)*50
					print gazeVecX, gazeVecY, (int(gazeVecX)+380), (int(gazeVecY)+250)

					my_agent.lastGazeVec_X = int(gazeVecX)
					my_agent.lastGazeVec_Y = int(gazeVecY)

 			if(SHOW_FACE_DETECTION):
				frame = cv2.resize(frame, (600, 480)) 
				xoff = 1300
				yoff = 600
				my_agent.startScreen[(yoff+my_agent.y_offset):(yoff+my_agent.y_offset+480), xoff:(xoff+600)] = frame  #order: y, x

			
		else: #if we are not using a camera, draw default eyes
			my_agent.lastGazeVec_X = 0
			my_agent.lastGazeVec_Y = 0

		my_agent.drawSelf()
		my_agent.drawScores()

		originalKeyPress = cv2.waitKey(10) #lets us check for special characters
		key= originalKeyPress & 0xFF       #convenience, to check for simple characters


		if(originalKeyPress==1114083):   #alternative way to end program
			if(my_agent.userName == "break"):
				break
		elif(originalKeyPress==1113938): #up arrow: this is our special key for debugging/to control+test interface
			#print "pressed up arrow"
			if(lastKeyPress ==  ord("q")): 
				print "Pressed quit"  #recommended way to end program
				print ''
				break
			elif(lastKeyPress ==  ord("s")):            #toggle face detection visualization
				if(SHOW_FACE_DETECTION==False):
					SHOW_FACE_DETECTION=True
				else:			
					SHOW_FACE_DETECTION=False 
					cv2.rectangle(my_agent.startScreen, (900, (450+my_agent.y_offset)), ((900+my_agent.face_width), (450+my_agent.y_offset+600)), (0, 0, 0), thickness=-1)
			elif(lastKeyPress ==  ord("t")): 
				if(SHOW_TIREDNESS==False):
					SHOW_TIREDNESS=True
				else:			
					SHOW_TIREDNESS=False 
			elif(lastKeyPress ==  ord("i")): 
				my_agent.agentValence+=0.1
				my_agent.moveAWeeBit(10,10)
				print "increased agentValence", my_agent.agentValence
			elif(lastKeyPress ==  ord("o")): 
				my_agent.agentValence-=0.1
				my_agent.moveAWeeBit(-10,-10)
				print "decreased agentValence", my_agent.agentValence

		elif(originalKeyPress==1113864):
			#print "Pressed backspace" #check there is something to delete, remove last char, and redraw
			if(my_agent.interactionState == 0 and my_agent.userName): 
				my_agent.userName = my_agent.userName[:-1]
				cv2.rectangle(my_agent.startScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
				cv2.putText(my_agent.startScreen, my_agent.userName, (30, 930), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

			if(my_agent.interactionState == 2 and my_agent.answerSnippet): 
				my_agent.answerSnippet = my_agent.answerSnippet[:-1]
				cv2.rectangle(my_agent.startScreen, (0, 860), (1024, 1080), (0, 0, 0), thickness=-1)
				cv2.putText(my_agent.startScreen, my_agent.userName, (30, 930), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

		elif(originalKeyPress==1048586):
			#print "Pressed return"
			if(my_agent.interactionState == 0): #in this state, the user is inputting their userid
				my_agent.userClickedInterface (1, 418, 702, 32) 

			if(my_agent.interactionState == 2): #in this state, the user is answering a quiz question
				my_agent.userClickedInterface (1, 418, 702, 32) 

		elif(key >=  ord(" ") and key <= ord("z")): 

			if(my_agent.asleepState == True):
				my_agent.asleepState = False

			my_agent.lastTimeUserDidSomething= datetime.now()

			if(my_agent.interactionState == 0 and ((key >=  ord("0") and key <= ord("9")) or (key >=  ord("A") and key <= ord("z"))) and len(my_agent.userName) < 10): #check length is okay
				my_agent.userName = my_agent.userName + str(unichr(key))
				cv2.putText(my_agent.startScreen, my_agent.userName, (30, 930), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

			if(my_agent.interactionState == 2 and len(my_agent.answerSnippet) < 20): #notice we permit various chars like spaces in the answers
				my_agent.answerSnippet = my_agent.answerSnippet + str(unichr(key))
				cv2.putText(my_agent.quizScreen, my_agent.answerSnippet, (30, 930), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
	
		if originalKeyPress !=-1:
			lastKeyPress = key
	cv2.destroyAllWindows()


if __name__== '__main__':

	print ''
    	print '-------------------------------------'
    	print '-       QUIZ GIVING AGENT           -'
    	print '-   JUN 2019, Martin at HPC (HH)    -'
    	print '-------------------------------------'
	print ''

	try:
		main()
	except rospy.ROSInterruptException:
		pass
	finally:
		pass



