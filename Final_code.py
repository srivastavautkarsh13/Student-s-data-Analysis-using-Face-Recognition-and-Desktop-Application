import sqlite3
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

LARGE_FONT= ("Caslon", 16)
SMALL_FONT= ("Caslon", 8)

class FaceRecognition(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        '''tk.Tk.iconbitmap(self, default="clienticon.ico")'''
        tk.Tk.wm_title(self, "FaceRecognition")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageSem1, PageSem2,PageSem3,PageSem4,PageSem5,PageSem6):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        
        def popupmsg(msg):
            popup = tk.Tk()
            popup.wm_title("!")
            label = ttk.Label(popup, text=msg, font=LARGE_FONT)
            label.place(x=110,y=20)
            B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
            B1.place(x=170,y=60)
            popup.geometry("400x150+450+250")
            popup.mainloop()
                        
            
        def recognizeface():
            import cv2
            import pickle
            import time

            face_cascade = cv2.CascadeClassifier('F:\Python\Open_CV\Face3/haarcascade_frontalface_alt2.xml')

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("F:\Python\Open_CV\Face3/face-trainner.yml")

            labels = {"person_name": 1}
            with open("F:\Python\Open_CV\Face3/face-labels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {v:k for k,v in og_labels.items()}
                
            start_time = time.time()
            capture_duration=10

            cap = cv2.VideoCapture(0)
            name='unknown'
            
            while(int(time.time() - start_time) < capture_duration):
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame=cv2.flip(frame,1)
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                	#print(x,y,w,h)
                    roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
                    roi_color = frame[y:y+h, x:x+w]
            
                	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
                    id_, conf = recognizer.predict(roi_gray)
                    if conf>=4 and conf <= 85:
                		#print(5: #id_)
                		#print(labels[id_])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name= labels[id_]
                        stroke = 2
                        cv2.putText(frame, name, (x+6,y-6), font, 1, (255,255,255), stroke, cv2.LINE_AA)
            
                    img_item = "1.png"
                    cv2.imwrite(img_item, roi_color)
                    
                    stroke = 2
                    end_cord_x = x + w
                    end_cord_y = y + h
                    cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0,0,255), stroke,cv2.FILLED)
                	#subitems = smile_cascade.detectMultiScale(roi_gray)
                	#for (ex,ey,ew,eh) in subitems:
                	#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                # Display the resulting frame
                cv2.imshow('frame',frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            if (name=="utkarsh"):
                controller.show_frame(PageOne)
            else:
                popupmsg("Unauthorised Access")
            
        
        '''def trainmodel():
            import cv2
            import os
            import numpy as np
            from PIL import Image
            import pickle
            os.chdir("F:\\Python\\Open_CV\\Face3")
            
            BASE_DIR = os.path.dirname(os.path.abspath('Faces_train.py'))
            image_dir = os.path.join(BASE_DIR, "images")
            
            face_cascade = cv2.CascadeClassifier('F:\Python\Open_CV\Face3/haarcascade_frontalface_alt2.xml')
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            current_id = 0
            label_ids = {}
            y_labels = []
            x_train = []
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.endswith("png") or file.endswith("jpg"):
                        path = os.path.join(root, file)
                        label = os.path.basename(root).replace(" ", "-").lower()
                    		#print(label, path)
                        if not label in label_ids:
                            label_ids[label] = current_id
                            current_id += 1
                        id_=label_ids[label]
                			#print(label_ids)
                			#y_labels.append(label) # some number
                			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                        pil_image = Image.open(path).convert("L") # grayscale
                        size = (550, 550)
                        final_image = pil_image.resize(size, Image.ANTIALIAS)
                        image_array = np.array(final_image, "uint8")
                			#print(image_array)
                        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                
                        for (x,y,w,h) in faces:
                            roi=image_array[y:y+h, x:x+w]
                            x_train.append(roi)
                            y_labels.append(id_)
            with open("face-labels.pickle", 'wb') as f:
                pickle.dump(label_ids, f)
            recognizer.train(x_train, np.array(y_labels))
            recognizer.save("face-trainner.yml")
            recognizeface()'''
        
        '''def recognizeface():
            import time
            import face_recognition
            import cv2
            import numpy as np
            import os
            os.chdir("F:\Python\Open_CV\Face3")
            
            video_capture = cv2.VideoCapture(0)
            
            name = "Unknown"
            
            # Load a sample picture and learn how to recognize it.
            heisenberg_image = face_recognition.load_image_file("Heisenberg.png")
            heisenberg_face_encoding = face_recognition.face_encodings(heisenberg_image)[0]
            
            jain_image = face_recognition.load_image_file("2.jpg")
            jain_face_encoding = face_recognition.face_encodings(jain_image)[0]
            
            
            # Load a second sample picture and learn how to recognize it.
            utkarsh_image = face_recognition.load_image_file("32.jpg")
            utkarsh_face_encoding = face_recognition.face_encodings(utkarsh_image)[0]
            
            # Create arrays of known face encodings and their names
            known_face_encodings = [
                heisenberg_face_encoding,
                jain_face_encoding,
                utkarsh_face_encoding
                
                
            ]
            known_face_names = [
                "Heisenberg",
                "Jain",
                "Utkarsh"
            ]
            
            # Initialize variables
            face_locations = []
            face_encodings = []
            face_names = []
            process_this_frame = True
            
            start_time = time.time()
            capture_duration=10
            
            while( int(time.time() - start_time) < capture_duration ):
                # Grab a single frame of video
                ret, frame = video_capture.read()
            
                # Resize and flip the frame of the video 
                frame=cv2.flip(frame,1)
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
                # Convert the image from BGR color (which OpenCV uses) to RGB color
                rgb_small_frame = small_frame[:, :, ::-1]
            
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        
            
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
            
                        face_names.append(name)
            
                process_this_frame = not process_this_frame
            
            
                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
            
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
                # Display the resulting image
                cv2.imshow('Video', frame)
                
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            
            video_capture.release()
            cv2.destroyAllWindows()
            
            print(name)
            
            if (name=="Utkarsh"):
                controller.show_frame(PageOne)
            else:
                popupmsg("Unauthorised Access")'''
                
                
                
        tk.Frame.__init__(self,parent)
        
        label = tk.Label(self, font=LARGE_FONT)
        label.pack()
        
        background_image=ImageTk.PhotoImage(file = "Heavy Rain.png")
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
    
        '''C = tk.Canvas(self, bg="blue", height=250, width=300)
        filename = ImageTk.PhotoImage(file = "Biometrics_eye.png")
        background_label = tk.Label(self, image=filename)
        
        label.background_label = filename
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        C.pack()
        background_image=ImageTk.PhotoImage(file = "bg2.jpg")
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)'''

        
        label1 = tk.Label(self, text="Welcome to the Face Recognizer App!!!",background='white', font=LARGE_FONT)
        label1.place(x=380,y=150)
        
        img3 = Image.open('images.jpg')
        img3 = img3.resize((750, 300), Image.ANTIALIAS)
        self.tkimage3 = ImageTk.PhotoImage(img3)
        tk.Label(self,image = self.tkimage3).place(x=200, y=200)
        
        button = ttk.Button(self, text="Recognize the FACE",command=recognizeface)
        button.place(x=520,y=515)
        
        label2 = tk.Label(self, text="Important Note : New window will open only for 5 sec., So please put your face infront of the WebCam properly. ",background='white', font=('Calibri',8))
        label2.place(x=310,y=573)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        name=tk.StringVar()
        mname=tk.StringVar()
        fname=tk.StringVar()
        rollno=tk.StringVar()
        status=tk.StringVar()
        course=tk.StringVar()
        course.set('I.T')
        
        def personal_info(): 
            conn = sqlite3.connect('fontydata.db')
            cursor = conn.execute("SELECT Student_Name,Mothers_Name,Fathers_Name,Roll_no,Category from Personal_info")
            for row in cursor:
                name.set(row[0])
                mname.set(row[1])
                fname.set(row[2])
                rollno.set(row[3])
                status.set(row[4])
            conn.close()
        personal_info()
                
        style = ttk.Style()
        style.configure('W.TButton', font =
               ('calibri', 10, 'bold', 'underline'), 
                foreground = 'red') 
        
        
        label = tk.Label(self, font=LARGE_FONT)
        
        background_image=ImageTk.PhotoImage(file = "Heavy Rain.png")
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        labelname = tk.Label(self, text="Name ", font=SMALL_FONT,background='grey',foreground='white',width=20)
        labelname.place(x=300,y=150)
        
        labelFname = tk.Label(self, text="Mother's Name ", font=SMALL_FONT,background='grey',foreground='white',width=20)
        labelFname.place(x=300,y=180)
        
        labelFname = tk.Label(self, text="Father's Name ", font=SMALL_FONT,background='grey',foreground='white',width=20)
        labelFname.place(x=300,y=210)
        
        labelrollno = tk.Label(self, text="Roll No. ", font=SMALL_FONT,background='grey',foreground='white',width=20)
        labelrollno.place(x=300,y=240)
        
        labelstatus = tk.Label(self, text="Status ", font=SMALL_FONT,background='grey',foreground='white',width=20)
        labelstatus.place(x=300,y=270)
        
        labelcourse = tk.Label(self, text="Course ", font=SMALL_FONT,background='grey',foreground='white',width=20)
        labelcourse.place(x=300,y=300)
        
        
        labelname2 = tk.Label(self, textvariable=name, font=SMALL_FONT,background='grey',foreground='white',width=40)
        labelname2.place(x=500,y=150)
        
        labelMname2 = tk.Label(self, textvariable=mname, font=SMALL_FONT,background='grey',foreground='white',width=40)
        labelMname2.place(x=500,y=180)
        
        labelFname2 = tk.Label(self, textvariable=fname, font=SMALL_FONT,background='grey',foreground='white',width=40)
        labelFname2.place(x=500,y=210)
        
        labelrollno2 = tk.Label(self, textvariable=rollno, font=SMALL_FONT,background='grey',foreground='white',width=40)
        labelrollno2.place(x=500,y=240)
        
        labelstatus2 = tk.Label(self, textvariable=status, font=SMALL_FONT,background='grey',foreground='white',width=40)
        labelstatus2.place(x=500,y=270)
        
        labelcourse2 = tk.Label(self, textvariable=course, font=SMALL_FONT,background='grey',foreground='white',width=40)
        labelcourse2.place(x=500,y=300)
        
        
        labelcourse2 = tk.Label(self, text='Choose the Semester -', font=SMALL_FONT,background='grey',foreground='white',width=73)
        labelcourse2.place(x=300,y=370)
        
        
        img3 = Image.open('utkarsh.jpg')
        img3 = img3.resize((150, 170), Image.ANTIALIAS)
        self.tkimage3 = ImageTk.PhotoImage(img3)
        tk.Label(self,image = self.tkimage3).place(x=800, y=150)
        
        r=tk.IntVar()
    
        def radioselected2():
            v=r.get()
            if(v==1):
                controller.show_frame(PageSem1)
            elif(v==2):
                controller.show_frame(PageSem2)
            elif(v==3):
                controller.show_frame(PageSem3)
            elif(v==4):
                controller.show_frame(PageSem4)
            elif(v==5):
                controller.show_frame(PageSem5)
            elif(v==6):
                controller.show_frame(PageSem6)
            
            
        labely1 = tk.Label(self, text=" Year 2016/2017 ", font=SMALL_FONT,background='grey',foreground='white',width=15)
        labely1.place(x=320,y=410)
        
        tk.Radiobutton(self, 
              text="Sem 1",
              variable=r,
              padx = 20, 
              value=1,).place(x=475,y=410)
        
        tk.Radiobutton(self, 
              text="Sem 2",
              variable=r,
              padx = 20, 
              value=2).place(x=620,y=410)
        
        labely2= tk.Label(self, text=" Year 2017/2018 ", font=SMALL_FONT,background='grey',foreground='white',width=15)
        labely2.place(x=320,y=450)
        
        tk.Radiobutton(self, 
              text="Sem 3",
              variable=r,
              padx = 20, 
              value=3).place(x=475,y=450)
        
        tk.Radiobutton(self, 
              text="Sem 4",
              variable=r,
              padx = 20, 
              value=4).place(x=620,y=450)
        
        labely3 = tk.Label(self, text=" Year 2018/2019 ", font=SMALL_FONT,background='grey',foreground='white',width=15)
        labely3.place(x=320,y=490)
        
        tk.Radiobutton(self, 
              text="Sem 5",
              variable=r,
              padx = 20, 
              value=5).place(x=475,y=490)
        
        tk.Radiobutton(self, 
              text="Sem 6",
              variable=r,
              padx = 20, 
              value=6).place(x=620,y=490)
        
        button1 = ttk.Button(self, text="PROCEED",
                            command=radioselected2)
        button1.place(x=490,y=530)
        
        button2 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button2.place(x=490,y=560)
        


class PageSem1(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        def donothing():
            print("hbcejhbce")
            
            
        def attendanceanalysis(subject):
            
            fig2 = Figure(figsize=(8.8,10))
            a2 = fig2.add_subplot(111)
            
            conn = sqlite3.connect('fontydata.db')            
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_1")
            counts_platform2 = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(2,5):
                        data=row[i]
                        counts_platform2.append(data)
                    break
                    
            a2.bar(range(len(counts_platform2)),counts_platform2,color="red",width=0.5)
            a2.set_xticks("Total_Classes","Classes_Held","Presents")
        
            canvas = FigureCanvasTkAgg(fig2, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
            conn.close()
            
        def marksanalysis(subject):
        
            fig1 = Figure(figsize=(8.8,10))
            a1 = fig1.add_subplot(111)
        
            conn = sqlite3.connect('fontydata.db') 
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_1")
        
        
            labels = 'ct1_marks', 'ct2_marks', 'Sem_marks', 'Internal_marks'
            counts_platform = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(5,9):
                        data=row[i]
                        counts_platform.append(data)
                    break
            
            a1.pie(counts_platform, labels=labels,shadow=True, startangle=90)
            a1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            a1.set_title(row[1])
            
            canvas = FigureCanvasTkAgg(fig1, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            conn.close()
            
        label = tk.Label(self, font=LARGE_FONT)
                         
        background_image = ImageTk.PhotoImage(file='Heavy Rain.png')
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        label_top2 = tk.Label(self, text="CHOOSE THE SUBJECT", font=LARGE_FONT,background='grey',foreground='white')
        label_top2.place(x=220,y=140)
            
        r=tk.IntVar()
        
        def radioselected1():
            v=r.get()
            if(v==1):
                marksanalysis('Engineering Maths-I')
            elif(v==2):
                marksanalysis('Engineering Physicss-I')
            elif(v==3):
                marksanalysis('Basic Electrical Engg')
            elif(v==4):
                marksanalysis('Professional Communication')
            elif(v==5):
                marksanalysis('Basic Electronics')
            elif(v==6):
                marksanalysis('Engg. Physics Lab')
            elif(v==7):
                marksanalysis('Basic Electrical Engg Lab')
            elif(v==8):
                marksanalysis('Professional Communication Lab')
            elif(v==9):
                marksanalysis('Workshop Practice')
        
        def radioselected2():
            v=r.get()
            if(v==1):
                attendanceanalysis('Engineering Maths-I')
            elif(v==2):
                attendanceanalysis('Engineering Physicss-I')
            elif(v==3):
                attendanceanalysis('Basic Electrical Engg')
            elif(v==4):
                attendanceanalysis('Professional Communication')
            elif(v==5):
                attendanceanalysis('Basic Electronics')
            elif(v==6):
                attendanceanalysis('Engg. Physics Lab')
            elif(v==7):
                attendanceanalysis('Basic Electrical Engg Lab')
            elif(v==8):
                attendanceanalysis('Professional Communication Lab')
            elif(v==9):
                attendanceanalysis('Workshop Practice')
                
        
        tk.Radiobutton(self, 
              text="Engineering Maths-I",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=1,).place(x=220,y=180)
        
        tk.Radiobutton(self, 
              text="Engineering Physicss-I",
              bg='grey',
              width=25,
              variable=r,
              padx = 20,
              anchor='w',
              value=2).place(x=220,y=220)
        
        tk.Radiobutton(self, 
              text="Basic Electrical Engg",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=3,).place(x=220,y=260)
        
        tk.Radiobutton(self, 
              text="Professional Communication",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=4).place(x=220,y=300)
        
        tk.Radiobutton(self, 
              text="Basic Electronics",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=5,).place(x=220,y=340)
        
        tk.Radiobutton(self, 
              text="Engg. Physics Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=6).place(x=220,y=380)
        
        tk.Radiobutton(self, 
              text="Basic Electrical Engg Lab",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=7).place(x=220,y=420)
        
        tk.Radiobutton(self, 
              text="Professional Communication Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=8).place(x=220,y=460)
        
        tk.Radiobutton(self, 
              text="Workshop Practice",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=9).place(x=220,y=500)
        
        label_top3 = tk.Label(self, text="CLICK HERE GET THE FINAL ANALYSIS REPORT", font=LARGE_FONT,background='grey',foreground='white')
        label_top3.place(x=520,y=200)
        
        button1 = ttk.Button(self, text="Marks Analysis",width=20,
                            command=radioselected1)
        button1.place(x=690,y=260)
        
        button2 = ttk.Button(self, text="Attendance Analysis",width=30,
                            command=radioselected2)
        button2.place(x=660,y=300)
        
        button3 = ttk.Button(self, text="BACK",
                            command=lambda: controller.show_frame(PageOne))
        button3.place(x=720,y=340)
        
        button4 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button4.place(x=720,y=380)
            
            
        
        '''def leftClick(event):
            print("Left")
        
        img1 = Image.open('engmaths.jpg')
        img1 = img1.resize((300, 200), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img1)
        img11 = tk.Label(self,image=img1, width=300, height=200)
        img11.image=img1
        img11.place(x=-0,y=0)
        
        button1 = ttk.Button(self, text="Marks",
                            command=lambda: EngineeringMaths1st('Engineering Maths-I'))
        button1.place(x=150,y=50)
        
        button2 = ttk.Button(self, text="Attendance",
                            command=lambda: EngineeringMaths1st('Engineering Maths-I'))
        button2.place(x=150,y=120)
        
        
        img2 = Image.open('engphy.jpg')
        img2 = img2.resize((300, 200), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img2)
        img22 = tk.Label(self,image=img2, width=300, height=200)
        img22.image=img2
        img22.place(x=304,y=0)
        
        button3 = ttk.Button(self, text="Marks",
                            command=lambda: EngineeringMaths1st('Engineering Maths-I'))
        button3.place(x=315,y=50)
        
        button4 = ttk.Button(self, text="Attendance",
                            command=lambda: EngineeringMaths1st('Engineering Maths-I'))
        button4.place(x=315,y=120)
        
        
        img3 = Image.open('engelec.jpg')
        img3 = img3.resize((300, 200), Image.ANTIALIAS)
        img3 = ImageTk.PhotoImage(img3)
        img33 = tk.Label(self,image=img3, width=300, height=200)
        img33.image=img3
        img33.place(x=608,y=0)
        
        button5 = ttk.Button(self, text="Marks",
                            command=lambda: EngineeringMaths1st('Engineering Maths-I'))
        button5.place(x=615,y=50)
        
        button6 = ttk.Button(self, text="Attendance",
                            command=lambda: EngineeringMaths1st('Engineering Maths-I'))
        button6.place(x=615,y=120)'''


class PageSem2(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        def donothing():
            print("hbcejhbce")
            
            
        def attendanceanalysis(subject):
            
            fig2 = Figure(figsize=(8.8,10))
            a2 = fig2.add_subplot(111)
            
            conn = sqlite3.connect('fontydata.db')            
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_2")
            columns=["Total_Classes","Classes_Held","Presents"]
            counts_platform2 = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(2,5):
                        data=row[i]
                        counts_platform2.append(data)
                    break
                    
            a2.bar(range(len(counts_platform2)),counts_platform2,color="red",width=0.5)
        
            canvas = FigureCanvasTkAgg(fig2, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
            conn.close()
            
        def marksanalysis(subject):
        
            fig1 = Figure(figsize=(8.8,10))
            a1 = fig1.add_subplot(111)
        
            conn = sqlite3.connect('fontydata.db') 
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_2")
        
        
            labels = 'ct1_marks', 'ct2_marks', 'Sem_marks', 'Internal_marks'
            counts_platform = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(5,9):
                        data=row[i]
                        counts_platform.append(data)
                    break
            
            a1.pie(counts_platform, labels=labels,shadow=True, startangle=90)
            a1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            a1.set_title(row[1])
            
            canvas = FigureCanvasTkAgg(fig1, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            conn.close()
            
        label = tk.Label(self, font=LARGE_FONT)
                         
        background_image = ImageTk.PhotoImage(file='Heavy Rain.png')
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        label_top2 = tk.Label(self, text="CHOOSE THE SUBJECT", font=LARGE_FONT,background='grey',foreground='white')
        label_top2.place(x=220,y=140)
            
        r=tk.IntVar()
        
        def radioselected1():
            v=r.get()
            if(v==1):
                marksanalysis('Engineering Maths-II')
            elif(v==2):
                marksanalysis('Engineering Physicss-II')
            elif(v==3):
                marksanalysis('Elements of Mechanical Engg')
            elif(v==4):
                marksanalysis('Computer System & Programming in C')
            elif(v==5):
                marksanalysis('Engineering Chemistry')
            elif(v==6):
                marksanalysis('Engg. Chemistry Lab')
            elif(v==7):
                marksanalysis('Elements of Mechanical Engg Lab')
            elif(v==8):
                marksanalysis('Professional Communication Lab')
            elif(v==9):
                marksanalysis('Computer Aided Engg. Graphics')
        
        def radioselected2():
            v=r.get()
            if(v==1):
                attendanceanalysis('Engineering Maths-II')
            elif(v==2):
                attendanceanalysis('Engineering Physicss-II')
            elif(v==3):
                attendanceanalysis('Elements of Mechanical Engg')
            elif(v==4):
                attendanceanalysis('Computer System & Programming in C')
            elif(v==5):
                attendanceanalysis('Engineering Chemistry')
            elif(v==6):
                attendanceanalysis('Engg. Chemistry Lab')
            elif(v==7):
                attendanceanalysis('Elements of Mechanical Engg Lab')
            elif(v==8):
                attendanceanalysis('Computer Progm. Lab')
            elif(v==9):
                attendanceanalysis('Computer Aided Engg. Graphics')
                
        
        tk.Radiobutton(self, 
              text="Engineering Maths-II",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=1,).place(x=220,y=180)
        
        tk.Radiobutton(self, 
              text="Engineering Physicss-II",
              bg='grey',
              width=25,
              variable=r,
              padx = 20,
              anchor='w',
              value=2).place(x=220,y=220)
        
        tk.Radiobutton(self, 
              text="Elements of Mechanical Engg",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=3,).place(x=220,y=260)
        
        tk.Radiobutton(self, 
              text="Computer System & Programming in C",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=4).place(x=220,y=300)
        
        tk.Radiobutton(self, 
              text="Engineering Chemistry",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=5,).place(x=220,y=340)
        
        tk.Radiobutton(self, 
              text="Engg. Chemistry Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=6).place(x=220,y=380)
        
        tk.Radiobutton(self, 
              text="Elements of Mechanical Engg Lab",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=7).place(x=220,y=420)
        
        tk.Radiobutton(self, 
              text="Computer Progm. Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=8).place(x=220,y=460)
        
        tk.Radiobutton(self, 
              text="Computer Aided Engg. Graphics",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=9).place(x=220,y=500)
        
        label_top3 = tk.Label(self, text="CLICK HERE GET THE FINAL ANALYSIS REPORT", font=LARGE_FONT,background='grey',foreground='white')
        label_top3.place(x=520,y=200)
        
        button1 = ttk.Button(self, text="Marks Analysis",width=20,
                            command=radioselected1)
        button1.place(x=690,y=260)
        
        button2 = ttk.Button(self, text="Attendance Analysis",width=30,
                            command=radioselected2)
        button2.place(x=660,y=300)
        
        button3 = ttk.Button(self, text="BACK",
                            command=lambda: controller.show_frame(PageOne))
        button3.place(x=720,y=340)
        
        button4 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button4.place(x=720,y=380)
        

class PageSem3(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        def donothing():
            print("hbcejhbce")
            
            
        def attendanceanalysis(subject):
            
            fig2 = Figure(figsize=(8.8,10))
            a2 = fig2.add_subplot(111)
            
            conn = sqlite3.connect('fontydata.db')            
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_3")
            columns=["Total_Classes","Classes_Held","Presents"]
            counts_platform2 = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(2,5):
                        data=row[i]
                        counts_platform2.append(data)
                    break
                    
            a2.bar(range(len(counts_platform2)),counts_platform2,color="red",width=0.5)
        
            canvas = FigureCanvasTkAgg(fig2, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
            conn.close()
            
        def marksanalysis(subject):
        
            fig1 = Figure(figsize=(8.8,10))
            a1 = fig1.add_subplot(111)
        
            conn = sqlite3.connect('fontydata.db') 
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_3")
        
        
            labels = 'ct1_marks', 'ct2_marks', 'Sem_marks', 'Internal_marks'
            counts_platform = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(5,9):
                        data=row[i]
                        counts_platform.append(data)
                    break
            
            a1.pie(counts_platform, labels=labels,shadow=True, startangle=90)
            a1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            a1.set_title(row[1])
            
            canvas = FigureCanvasTkAgg(fig1, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            conn.close()
            
        label = tk.Label(self, font=LARGE_FONT)
                         
        background_image = ImageTk.PhotoImage(file='Heavy Rain.png')
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        label_top2 = tk.Label(self, text="CHOOSE THE SUBJECT", font=LARGE_FONT,background='grey',foreground='white')
        label_top2.place(x=220,y=140)
            
        r=tk.IntVar()
        
        def radioselected1():
            v=r.get()
            if(v==1):
                marksanalysis('Mathematics-III')
            elif(v==2):
                marksanalysis('Universal Human Values & Professional Ethics')
            elif(v==3):
                marksanalysis('Digital Logic Design')
            elif(v==4):
                marksanalysis('Discrete Structures & Theory of Logic')
            elif(v==5):
                marksanalysis('Computer Organization and Architecture')
            elif(v==6):
                marksanalysis('Data Structures')
            elif(v==7):
                marksanalysis('Digital Logic Design Lab')
            elif(v==8):
                marksanalysis('Discrete Structure & Logic Lab')
            elif(v==9):
                marksanalysis('Computer Organization Lab')
            elif(v==10):
                marksanalysis('Data Structures Using C')
        
        def radioselected2():
            v=r.get()
            if(v==1):
                attendanceanalysis('Mathematics-III')
            elif(v==2):
                attendanceanalysis('Universal Human Values & Professional Ethics')
            elif(v==3):
                attendanceanalysis('Digital Logic Design')
            elif(v==4):
                attendanceanalysis('Discrete Structures & Theory of Logic')
            elif(v==5):
                attendanceanalysis('Computer Organization and Architecture')
            elif(v==6):
                attendanceanalysis('Data Structures')
            elif(v==7):
                attendanceanalysis('Digital Logic Design Lab')
            elif(v==8):
                attendanceanalysis('Discrete Structure & Logic Lab')
            elif(v==9):
                attendanceanalysis('Computer Organization Lab')
            elif(v==10):
                attendanceanalysis('Data Structures Using C')
                
        
        tk.Radiobutton(self, 
              text="Mathematics-III",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=1,).place(x=220,y=180)
        
        tk.Radiobutton(self, 
              text="Universal Human Values & Professional Ethics",
              bg='grey',
              width=25,
              variable=r,
              padx = 20,
              anchor='w',
              value=2).place(x=220,y=220)
        
        tk.Radiobutton(self, 
              text="Digital Logic Design",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=3,).place(x=220,y=260)
        
        tk.Radiobutton(self, 
              text="Discrete Structures & Theory of Logic",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=4).place(x=220,y=300)
        
        tk.Radiobutton(self, 
              text="Computer Organization and Architecture",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=5,).place(x=220,y=340)
        
        tk.Radiobutton(self, 
              text="Data Structures",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=6).place(x=220,y=380)
        
        tk.Radiobutton(self, 
              text="Digital Logic Design Lab",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=7).place(x=220,y=420)
        
        tk.Radiobutton(self, 
              text="Discrete Structure & Logic Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=8).place(x=220,y=460)
        
        tk.Radiobutton(self, 
              text="Computer Organization Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=9).place(x=220,y=500)
        
        tk.Radiobutton(self, 
              text="Data Structures Using C",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=10).place(x=220,y=540)
        
        label_top3 = tk.Label(self, text="CLICK HERE GET THE FINAL ANALYSIS REPORT", font=LARGE_FONT,background='grey',foreground='white')
        label_top3.place(x=520,y=200)
        
        button1 = ttk.Button(self, text="Marks Analysis",width=20,
                            command=radioselected1)
        button1.place(x=690,y=260)
        
        button2 = ttk.Button(self, text="Attendance Analysis",width=30,
                            command=radioselected2)
        button2.place(x=660,y=300)
        
        button3 = ttk.Button(self, text="BACK",
                            command=lambda: controller.show_frame(PageOne))
        button3.place(x=720,y=340)
        
        button4 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button4.place(x=720,y=380)


class PageSem4(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        def donothing():
            print("hbcejhbce")
            
            
        def attendanceanalysis(subject):
            
            fig2 = Figure(figsize=(8.8,10))
            a2 = fig2.add_subplot(111)
            
            conn = sqlite3.connect('fontydata.db')            
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_4")
            columns=["Total_Classes","Classes_Held","Presents"]
            counts_platform2 = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(2,5):
                        data=row[i]
                        counts_platform2.append(data)
                    break
                    
            a2.bar(range(len(counts_platform2)),counts_platform2,color="red",width=0.5)
        
            canvas = FigureCanvasTkAgg(fig2, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
            conn.close()
            
        def marksanalysis(subject):
        
            fig1 = Figure(figsize=(8.8,10))
            a1 = fig1.add_subplot(111)
        
            conn = sqlite3.connect('fontydata.db') 
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_4")
        
        
            labels = 'ct1_marks', 'ct2_marks', 'Sem_marks', 'Internal_marks'
            counts_platform = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(5,9):
                        data=row[i]
                        counts_platform.append(data)
                    break
            
            a1.pie(counts_platform, labels=labels,shadow=True, startangle=90)
            a1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            a1.set_title(row[1])
            
            canvas = FigureCanvasTkAgg(fig1, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            conn.close()
            
        label = tk.Label(self, font=LARGE_FONT)
                         
        background_image = ImageTk.PhotoImage(file='Heavy Rain.png')
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        label_top2 = tk.Label(self, text="CHOOSE THE SUBJECT", font=LARGE_FONT,background='grey',foreground='white')
        label_top2.place(x=220,y=140)
            
        r=tk.IntVar()
        
        def radioselected1():
            v=r.get()
            if(v==1):
                marksanalysis('Science Based OE')
            elif(v==2):
                marksanalysis('Environment & Ecology')
            elif(v==3):
                marksanalysis('Information Theory and Coding')
            elif(v==4):
                marksanalysis('Operating Systems')
            elif(v==5):
                marksanalysis('Software Engineering')
            elif(v==6):
                marksanalysis('Theory of Automata and Formal Languages')
            elif(v==7):
                marksanalysis('Operating Systems Lab')
            elif(v==8):
                marksanalysis('Software Engineering Lab')
            elif(v==9):
                marksanalysis('TAFL Lab')
            elif(v==10):
                marksanalysis('Python Language Programming Lab')
        
        def radioselected2():
            v=r.get()
            if(v==1):
                attendanceanalysis('Science Based OE')
            elif(v==2):
                attendanceanalysis('Environment & Ecology')
            elif(v==3):
                attendanceanalysis('Information Theory and Coding')
            elif(v==4):
                attendanceanalysis('Operating Systems')
            elif(v==5):
                attendanceanalysis('Software Engineering')
            elif(v==6):
                attendanceanalysis('Theory of Automata and Formal Languages')
            elif(v==7):
                attendanceanalysis('Operating Systems Lab')
            elif(v==8):
                attendanceanalysis('Software Engineering Lab')
            elif(v==9):
                attendanceanalysis('TAFL Lab')
            elif(v==10):
                attendanceanalysis('Python Language Programming Lab')
                
        
        tk.Radiobutton(self, 
              text="Science Based OE",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=1,).place(x=220,y=180)
        
        tk.Radiobutton(self, 
              text="Environment & Ecology",
              bg='grey',
              width=25,
              variable=r,
              padx = 20,
              anchor='w',
              value=2).place(x=220,y=220)
        
        tk.Radiobutton(self, 
              text="Information Theory and Coding",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=3,).place(x=220,y=260)
        
        tk.Radiobutton(self, 
              text="Operating Systems",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=4).place(x=220,y=300)
        
        tk.Radiobutton(self, 
              text="Software Engineering",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=5,).place(x=220,y=340)
        
        tk.Radiobutton(self, 
              text="Theory of Automata and Formal Languages",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=6).place(x=220,y=380)
        
        tk.Radiobutton(self, 
              text="Operating Systems Lab",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=7).place(x=220,y=420)
        
        tk.Radiobutton(self, 
              text="Software Engineering Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=8).place(x=220,y=460)
        
        tk.Radiobutton(self, 
              text="TAFL Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=9).place(x=220,y=500)
    
        tk.Radiobutton(self, 
              text="Python Language Programming Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=10).place(x=220,y=540)
        
        label_top3 = tk.Label(self, text="CLICK HERE GET THE FINAL ANALYSIS REPORT", font=LARGE_FONT,background='grey',foreground='white')
        label_top3.place(x=520,y=200)
        
        button1 = ttk.Button(self, text="Marks Analysis",width=20,
                            command=radioselected1)
        button1.place(x=690,y=260)
        
        button2 = ttk.Button(self, text="Attendance Analysis",width=30,
                            command=radioselected2)
        button2.place(x=660,y=300)
        
        button3 = ttk.Button(self, text="BACK",
                            command=lambda: controller.show_frame(PageOne))
        button3.place(x=720,y=340)
        
        button4 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button4.place(x=720,y=380)
        
class PageSem5(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        def donothing():
            print("hbcejhbce")
            
            
        def attendanceanalysis(subject):
            
            fig2 = Figure(figsize=(8.8,10))
            a2 = fig2.add_subplot(111)
            
            conn = sqlite3.connect('fontydata.db')            
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_5")
            columns=["Total_Classes","Classes_Held","Presents"]
            counts_platform2 = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(2,5):
                        data=row[i]
                        counts_platform2.append(data)
                    break
                    
            a2.bar(range(len(counts_platform2)),counts_platform2,color="red",width=0.5)
        
            canvas = FigureCanvasTkAgg(fig2, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
            conn.close()
            
        def marksanalysis(subject):
        
            fig1 = Figure(figsize=(8.8,10))
            a1 = fig1.add_subplot(111)
        
            conn = sqlite3.connect('fontydata.db') 
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_5")
        
        
            labels = 'ct1_marks', 'ct2_marks', 'Sem_marks', 'Internal_marks'
            counts_platform = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(5,9):
                        data=row[i]
                        counts_platform.append(data)
                    break
            
            a1.pie(counts_platform, labels=labels,shadow=True, startangle=90)
            a1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            a1.set_title(row[1])
            
            canvas = FigureCanvasTkAgg(fig1, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            conn.close()
            
        label = tk.Label(self, font=LARGE_FONT)
                         
        background_image = ImageTk.PhotoImage(file='Heavy Rain.png')
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        label_top2 = tk.Label(self, text="CHOOSE THE SUBJECT", font=LARGE_FONT,background='grey',foreground='white')
        label_top2.place(x=220,y=140)
            
        r=tk.IntVar()
        
        def radioselected1():
            v=r.get()
            if(v==1):
                marksanalysis('Managerial Economics')
            elif(v==2):
                marksanalysis('Cyber Security')
            elif(v==3):
                marksanalysis('Database Management Sysytem')
            elif(v==4):
                marksanalysis('Design and Analysis of Algorithms')
            elif(v==5):
                marksanalysis('Principles of Programming Language')
            elif(v==6):
                marksanalysis('Object Oriented Programming')
            elif(v==7):
                marksanalysis('Database Management Systems Lab')
            elif(v==8):
                marksanalysis('Design and Analysis of Algorithm Lab')
            elif(v==9):
                marksanalysis('Principles of Programming Languages Lab')
            elif(v==10):
                marksanalysis('Object Oriented Techniques Lab')
        
        def radioselected2():
            v=r.get()
            if(v==1):
                attendanceanalysis('Managerial Economics')
            elif(v==2):
                attendanceanalysis('Cyber Security')
            elif(v==3):
                attendanceanalysis('Database Management Sysytem')
            elif(v==4):
                attendanceanalysis('Design and Analysis of Algorithms')
            elif(v==5):
                attendanceanalysis('Principles of Programming Language')
            elif(v==6):
                attendanceanalysis('Object Oriented Programming')
            elif(v==7):
                attendanceanalysis('Database Management Systems Lab')
            elif(v==8):
                attendanceanalysis('Design and Analysis of Algorithm Lab')
            elif(v==9):
                attendanceanalysis('Principles of Programming Languages Lab')
            elif(v==10):
                attendanceanalysis('Object Oriented Techniques Lab')
                
        
        tk.Radiobutton(self, 
              text="Managerial Economics",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=1,).place(x=220,y=180)
        
        tk.Radiobutton(self, 
              text="Cyber Security",
              bg='grey',
              width=25,
              variable=r,
              padx = 20,
              anchor='w',
              value=2).place(x=220,y=220)
        
        tk.Radiobutton(self, 
              text="Database Management Sysytem",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=3,).place(x=220,y=260)
        
        tk.Radiobutton(self, 
              text="Design and Analysis of Algorithms",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=4).place(x=220,y=300)
        
        tk.Radiobutton(self, 
              text="Principles of Programming Language",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=5,).place(x=220,y=340)
        
        tk.Radiobutton(self, 
              text="Object Oriented Programming",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=6).place(x=220,y=380)
        
        tk.Radiobutton(self, 
              text="Database Management Systems Lab",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=7).place(x=220,y=420)
        
        tk.Radiobutton(self, 
              text="Design and Analysis of Algorithm Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=8).place(x=220,y=460)
        
        tk.Radiobutton(self, 
              text="Principles of Programming Languages Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=9).place(x=220,y=500)
    
        tk.Radiobutton(self, 
              text="Object Oriented Techniques Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=10).place(x=220,y=540)
        
        label_top3 = tk.Label(self, text="CLICK HERE GET THE FINAL ANALYSIS REPORT", font=LARGE_FONT,background='grey',foreground='white')
        label_top3.place(x=520,y=200)
        
        button1 = ttk.Button(self, text="Marks Analysis",width=20,
                            command=radioselected1)
        button1.place(x=690,y=260)
        
        button2 = ttk.Button(self, text="Attendance Analysis",width=30,
                            command=radioselected2)
        button2.place(x=660,y=300)
        
        button3 = ttk.Button(self, text="BACK",
                            command=lambda: controller.show_frame(PageOne))
        button3.place(x=720,y=340)
        
        button4 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button4.place(x=720,y=380)
        

class PageSem6(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        def donothing():
            print("hbcejhbce")
            
            
        def attendanceanalysis(subject):
            
            fig2 = Figure(figsize=(8.8,10))
            a2 = fig2.add_subplot(111)
            
            conn = sqlite3.connect('fontydata.db')            
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_6")
            columns=["Total_Classes","Classes_Held","Presents"]
            counts_platform2 = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(2,5):
                        data=row[i]
                        counts_platform2.append(data)
                    break
                    
            a2.bar(range(len(counts_platform2)),counts_platform2,color="red",width=0.5)
        
            canvas = FigureCanvasTkAgg(fig2, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
            conn.close()
            
        def marksanalysis(subject):
        
            fig1 = Figure(figsize=(8.8,10))
            a1 = fig1.add_subplot(111)
        
            conn = sqlite3.connect('fontydata.db') 
            cursor = conn.execute("SELECT Subject_Code,Subject_Name,Total_Classes,Classes_Held,Presents,ct1_marks,ct2_marks,Sem_marks,Internal_marks from Semester_6")
        
        
            labels = 'ct1_marks', 'ct2_marks', 'Sem_marks', 'Internal_marks'
            counts_platform = []
            for row in cursor:
                if(row[1]==subject):
                    for i in range(5,9):
                        data=row[i]
                        counts_platform.append(data)
                    break
            
            a1.pie(counts_platform, labels=labels,shadow=True, startangle=90)
            a1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            a1.set_title(row[1])
            
            canvas = FigureCanvasTkAgg(fig1, self)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2TkAgg(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            conn.close()
            
        label = tk.Label(self, font=LARGE_FONT)
                         
        background_image = ImageTk.PhotoImage(file='Heavy Rain.png')
        background_label = tk.Label(self, image=background_image)
        label.background_label=background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        label_top = tk.Label(self, text="Institute of Engineering & Technology, Lucknow \n अभियांत्रिकी एवं प्रौद्योगिकी संस्थान, लखनऊ \n An Autonomous Constituent Institute of Dr. A.P.J. Abdul Kalam Technical University, U.P., Lucknow", font=LARGE_FONT,background='grey',foreground='white')
        label_top.place(x=106,y=20)
        
        img = Image.open('ietlogo1.png')
        img = img.resize((92, 97), Image.ANTIALIAS)
        self.tkimage = ImageTk.PhotoImage(img)
        tk.Label(self,image = self.tkimage).place(x=5, y=8)
        
        img2 = Image.open('logo.png')
        img2 = img2.resize((92, 97), Image.ANTIALIAS)
        self.tkimage2 = ImageTk.PhotoImage(img2)
        tk.Label(self,image = self.tkimage2).place(x=1043, y=5)
        
        label_top2 = tk.Label(self, text="CHOOSE THE SUBJECT", font=LARGE_FONT,background='grey',foreground='white')
        label_top2.place(x=220,y=140)
            
        r=tk.IntVar()
        
        def radioselected1():
            v=r.get()
            if(v==1):
                marksanalysis('Industrial Management')
            elif(v==2):
                marksanalysis('Industrial Sociology')
            elif(v==3):
                marksanalysis('Computer Networks')
            elif(v==4):
                marksanalysis('Compiler Design')
            elif(v==5):
                marksanalysis('Web Technology')
            elif(v==6):
                marksanalysis('Data Warehousing & Data Mining')
            elif(v==7):
                marksanalysis('Computer Networks Lab')
            elif(v==8):
                marksanalysis('Compiler Design Lab')
            elif(v==9):
                marksanalysis('Web Technology Lab')
            elif(v==10):
                marksanalysis('Data Warehousing & Data Mining Lab')
        
        def radioselected2():
            v=r.get()
            if(v==1):
                attendanceanalysis('Industrial Management')
            elif(v==2):
                attendanceanalysis('Industrial Sociology')
            elif(v==3):
                attendanceanalysis('Computer Networks')
            elif(v==4):
                attendanceanalysis('Compiler Design')
            elif(v==5):
                attendanceanalysis('Web Technology')
            elif(v==6):
                attendanceanalysis('Data Warehousing & Data Mining')
            elif(v==7):
                attendanceanalysis('Computer Networks Lab')
            elif(v==8):
                attendanceanalysis('Compiler Design Lab')
            elif(v==9):
                attendanceanalysis('Web Technology Lab')
            elif(v==10):
                attendanceanalysis('Data Warehousing & Data Mining Lab')
                
        
        tk.Radiobutton(self, 
              text="Industrial Management",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=1,).place(x=220,y=180)
        
        tk.Radiobutton(self, 
              text="Industrial Sociology",
              bg='grey',
              width=25,
              variable=r,
              padx = 20,
              anchor='w',
              value=2).place(x=220,y=220)
        
        tk.Radiobutton(self, 
              text="Computer Networks",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=3,).place(x=220,y=260)
        
        tk.Radiobutton(self, 
              text="Compiler Design",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=4).place(x=220,y=300)
        
        tk.Radiobutton(self, 
              text="Web Technology",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=5,).place(x=220,y=340)
        
        tk.Radiobutton(self, 
              text="Data Warehousing & Data Mining",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=6).place(x=220,y=380)
        
        tk.Radiobutton(self, 
              text="Computer Networks Lab",
              width=25,
              bg='grey',
              variable=r,
              padx = 20, 
              anchor='w',
              value=7).place(x=220,y=420)
        
        tk.Radiobutton(self, 
              text="Compiler Design Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=8).place(x=220,y=460)
        
        tk.Radiobutton(self, 
              text="Web Technology Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=9).place(x=220,y=500)
        
        tk.Radiobutton(self, 
              text="Data Warehousing & Data Mining Lab",
              bg='grey',
              width=25,
              variable=r,
              padx = 20, 
              anchor='w',
              value=10).place(x=220,y=540)
        
        label_top3 = tk.Label(self, text="CLICK HERE GET THE FINAL ANALYSIS REPORT", font=LARGE_FONT,background='grey',foreground='white')
        label_top3.place(x=520,y=200)
        
        button1 = ttk.Button(self, text="Marks Analysis",width=20,
                            command=radioselected1)
        button1.place(x=690,y=260)
        
        button2 = ttk.Button(self, text="Attendance Analysis",width=30,
                            command=radioselected2)
        button2.place(x=660,y=300)
        
        button3 = ttk.Button(self, text="BACK",
                            command=lambda: controller.show_frame(PageOne))
        button3.place(x=720,y=340)
        
        button4 = ttk.Button(self, text="HOME",
                            command=lambda: controller.show_frame(StartPage))
        button4.place(x=720,y=380)

app = FaceRecognition()
app.geometry("1150x600+100+50")
app.mainloop()