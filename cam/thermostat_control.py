import cv2
import requests
import time

# Constants
RTSP_URL = "rtsp://192.168.43.1:8080/h264_opus.sdp"
THERMOSTAT_URL_ON = "http://192.168.43.100/aircon/set_control_info?pow=1&mode=1&stemp=25&f_rate=3&f_dir=0"
THERMOSTAT_URL_OFF = "http://192.168.43.100/aircon/set_control_info?pow=0"

# Load pre-trained Haar Cascade classifier for people detection
people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Function to control the thermostat
def control_thermostat(power_on):
    if power_on:
        requests.get(THERMOSTAT_URL_ON)
        print("Thermostat turned ON")
    else:
        requests.get(THERMOSTAT_URL_OFF)
        print("Thermostat turned OFF")

# Start video capture
cap = cv2.VideoCapture(RTSP_URL)

# State variables
thermostat_on = False
last_detection_time = time.time()
detection_timeout = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the frame
    people = people_cascade.detectMultiScale(gray, 1.1, 4)

    # Check if people are detected
    if len(people) > 0:
        if not thermostat_on:
            control_thermostat(True)
            thermostat_on = True
        last_detection_time = time.time()  # Reset the timer
    else:
        # Check if the timeout has been reached
        if thermostat_on and (time.time() - last_detection_time > detection_timeout):
            control_thermostat(False)
            thermostat_on = False

    # Display the video stream (optional)
    cv2.imshow('CCTV Stream', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
