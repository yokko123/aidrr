import cv2
import datetime
import pandas
import matplotlib.pyplot as plt

first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End", "Duration"])
video = cv2.VideoCapture("test3.mp4")

no_of_frames = 1

while True:
    no_of_frames = no_of_frames + 1
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    print(frame)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    th_delta = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    th_delta = cv2.dilate(th_delta,None, iterations=0)
    (cnts, _) = cv2.findContours(th_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 5000:
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[-1] ==1 and status_list[-2] == 0:
        times.append(datetime.datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.datetime.now())

    cv2.imshow('Capturing', frame)
    cv2.imshow('Delta Frame', delta_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i+1], "Duration": times[i+1]-times[i]}, ignore_index=True)

df.to_csv("Motion_Times"+".csv")
video.release()
cv2.destroyAllWindows()