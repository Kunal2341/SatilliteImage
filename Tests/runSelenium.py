import streamlit as st
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
import pandas as pd
from IPython.core.display import HTML
import time
from selenium import webdriver
from googlemaps import Client as GoogleMaps
global KEY
import base64
import cv2
from datetime import timedelta
import pandas as pd
import datetime
import pyautogui as ms
import time
import pygetwindow
import math
from PIL import Image, ImageDraw

def scale_to_im(x, a=0, b=255):
    # TODO: Implement Min-Max scaling for grayscale image data
    ma = (np.max(x))
    if (ma == 0):
        return x.astype(np.uint8)
    mi = (np.min(x))
    normalized_data = ((x.astype(np.float) - float(mi)) / float(ma))  # normalize [0-1]
    normalized_data = (normalized_data * b + a * (1 - normalized_data))  # Scale values here
    return normalized_data.astype(np.uint8)
def nothing(x):
    pass
def channels3(x):
    # Stack grayscale images together to increase the color channels to 3
    return np.dstack((x, x, x))
def sidebyside(x, y):
    # Concatenate images side by side (horizontally)
    return np.concatenate((x, y), axis=1)
def updown(x, y):
    # Concatenate images up and down (vertically)
    return np.concatenate((x, y), axis=0)
def extractLargerSegment(maskROAD):
    thresh = maskROAD
    # _, contours, hierarchy = cv2.findContours(maskROAD.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxA = 0
    maskTemp = np.zeros_like(maskROAD)

    if (len(contours) > 0):
        for h, cnt in enumerate(contours):
            if (cv2.contourArea(cnt) > maxA):
                cntMax = cnt
                maxA = cv2.contourArea(cnt)
        mask = np.zeros(maskROAD.shape, np.uint8)
        cv2.drawContours(maskTemp, [cntMax], 0, 255, -1)
        maskROAD = cv2.bitwise_and(maskROAD, maskTemp)
    return maskROAD
def post_process(img):
    kernel = np.ones((5, 5), np.uint8)
    img_out = cv2.erode(img, kernel, iterations=3)
    kernel = np.ones((20, 20), np.uint8)
    img_out = cv2.dilate(img_out, kernel, iterations=5)

    img_out = extractLargerSegment(img_out)

    return img_out
def display(img_init, img_hsv, img_out2, img_out):
    mask = scale_to_im(np.dstack((img_out, np.zeros_like(img_out), np.zeros_like(img_out))))
    cv2.imshow('Output', updown(sidebyside(cv2.addWeighted(img_init, 1, mask, 0.3, 0), img_hsv),
                                sidebyside(channels3(img_out), channels3(img_out2))))
def detectionProcess(frame, model, winH=32, winW=32, depth=1, nb_images=2, scale=1.2, stepSize=10, thres_score=0):
    index = 0
    totalWindows = 0
    correct = 0

    bbox_list = []
    score = []

    for resized in pyramid(frame, scale=scale, minSize=(winH, winW), nb_images=nb_images):
        # gray = cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY)
        # loop over the sliding window for each layer of the pyramid
        scale = frame.shape[0] / resized.shape[0]
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            if (depth == 1):
                window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                window = np.expand_dims(window, 3)

            window = window[None, :, :, :]

            totalWindows += 1

            class_out = model.predict((window.astype(np.float32)) / 255., batch_size=1)[0]

            if (class_out < thres_score):
                bbox_list.append(((int(x * scale)), int(y * scale), int((x + winW) * scale), int((y + winH) * scale)))
                score.append(class_out)
                correct += 1

        index += 1

    return bbox_list, totalWindows, correct, score
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def pyramid(image, scale=1.5, minSize=(30, 30), nb_images=3):
    # yield the original image
    yield image
    count = 0

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)

        image = cv2.resize(image, (w, h))
        count += 1
        scale = np.power((1 / scale), count)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0] or (count == nb_images):
            break

        # yield the next image in the pyramid
        yield image
def drawBoxes(frame, bbox_list):
    for i in range(len(bbox_list)):
        box = bbox_list[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

    return frame


def addressToLatLong(addy):
    gmaps = GoogleMaps(KEY)
    geocode_result = gmaps.geocode(addy)
    x = geocode_result[0]['geometry']['location']['lat']
    y = geocode_result[0]['geometry']['location']['lng']
    return x,y
def getLink(lat,long,zoom):
    return ("https://www.google.com/maps/@" + str(lat) + "," + str(long) + "," + str(zoom) + "m/data=!3m1!1e3")
def getImg(link):
    url = link
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    op.add_argument("--start-maximized")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)
    driver.get(url)
    time.sleep(1)
    closedSideBar = False
    while (not closedSideBar):
        try:
            driver.find_element_by_xpath(
                "/html/body/jsl/div[3]/div[9]/div[3]/div[3]/div/div[2]/div/div[1]/div/button").click()
            closedSideBar = True
        except:
            time.sleep(0.7)
            closedSideBar = False
    takePic = driver.find_element_by_tag_name('body')
    #imgName =
    run = takePic.screenshot("test.png")
    driver.quit()
    return run,
def showMap(lat,long):
    map_data = pd.DataFrame({'lat': [lat], 'lon': [long]})
    st.map(map_data, zoom=16)

KEY = "AIzaSyCMWYThmY65PS2t6pg3lKLTzomkxkXTV0o"




#totalGetData = st.beta_expander("Get Data")
st.header("Input Data")
#---------------------------------------------------------------------------------------------
#st.header("Duration")
duration = st.beta_expander("Duration")

startDateTimeCol1, endDatetimeCol2 = duration.beta_columns(2)
startDateTimeCol1.header("Start")
endDatetimeCol2.header("End")
startDate = startDateTimeCol1.date_input("Start date",datetime.date(datetime.datetime.now().year, datetime.datetime.now().month-3,
                                                                    datetime.datetime.now().day))
#startTime = startDateTimeCol1.time_input('Start time', datetime.time(8, 45))

if startDate.year < 1985:
    duration.error("Please choose earlier date")

if startDate.month < 10:
    Startmonth = "0"+ str(startDate.month)
elif startDate.month >10 and startDate.month <= 12:
    Startmonth = str(startDate.month)

if startDate.day < 10:
    Startday = "0"+ str(startDate.day)
elif startDate.day >10 and startDate.day <= 31:
    Startday = str(startDate.day)

finalDateStart = Startmonth + "/" + Startday + "/" + str(startDate.year)[2:4] + " "
#if startTime.hour > 12:
    #finalTimeStart = str(startTime.hour-12) + ":" + str(startTime.minute) + " PM"
#if startTime.hour < 12:
    #finalTimeStart = str(startTime.hour) + ":" + str(startTime.minute) + " AM"
finalDateTimeStart = finalDateStart#+finalTimeStart

#-------------------------------------------------------------------------------------------

endDate = endDatetimeCol2.date_input("End date",datetime.date(datetime.datetime.now().year, datetime.datetime.now().month,
                                                              datetime.datetime.now().day))
#endTime = endDatetimeCol2.time_input('End time', datetime.time(datetime.datetime.now().hour, datetime.datetime.now().minute))

if endDate.year < 1985:
    duration.error("Please choose earlier date")
if (endDate - startDate).days == 0: #and (endTime.hour - startTime.hour) == 0 and (endTime.minute - startTime.minute) == 0:
    duration.error("Please choose different dates or times")


if endDate.month < 10:
    Startmonth = "0"+ str(endDate.month)
elif endDate.month >10 and endDate.month <= 12:
    Startmonth = str(endDate.month)

if endDate.day < 10:
    Startday = "0"+ str(endDate.day)
elif endDate.day >10 and endDate.day <= 31:
    Startday = str(endDate.day)

finalDateEnd = Startmonth + "/" + Startday + "/" + str(endDate.year)[2:4] + " "
#if endTime.hour > 12:
    #finalTimeEnd = str(endTime.hour-12) + ":" + str(endTime.minute) + " PM"
#if endTime.hour < 12:
    #finalTimeEnd = str(endTime.hour) + ":" + str(endTime.minute) + " AM"
finalDateTimeEnd = finalDateEnd# + finalTimeEnd

#---------------------------------------------------------------------------------------------
#st.header("Interval")
interval = st.beta_expander("Interval")

intervalCol1, intervalCol2 = interval.beta_columns(2)
intervalCol1.header("Interval Value")
intervalValue = intervalCol1.number_input("", value=1,step = 1)
intervalCol2.header("Interval Choice")
#intervalChoice = intervalCol2.radio("", ("Hour", "Day", "Year"))
intervalChoice = intervalCol2.radio("", ("Month", "Year"))

#if (intervalChoice == "Hour" and (intervalValue > 23 or intervalValue < 0)):
    #interval.error("ERROR: Choose correct hour time which is less than " + str(23) + " hours and greater " + str(0) + " hour")

#Fix Error here ------>
#if (intervalChoice == "Month" and (intervalValue > ((endDate - startDate).days/30) or intervalValue < 0)):
    #interval.error("ERROR: Choose correct day which is less than " + str(round(((endDate - startDate).days/30),2)) + " months and greater " +  str(0) + " day")
#if ((((endDate - startDate).days)/365 < 1) and intervalChoice == "Year"):
    #interval.error("ERROR: Choose a larger time duration if you want a yearly interval")


def findEveryDateTime(sDate, sTime, eDate, eTime, iChoice, iValue):
    startDateTime = datetime.datetime.strptime(str(sDate) + "|" + str(sTime), '%Y-%m-%d|%H:%M:%S')
    endDateTime = datetime.datetime.strptime(str(eDate) + "|" + str(eTime), '%Y-%m-%d|%H:%M:%S')

    lstDateTimes = [startDateTime.strftime('%Y-%m-%d|%H:%M:%S')]

    startTime = startDateTime

    if iChoice == "Month":
        while startTime < endDateTime:
            startTime += timedelta(days=iValue*30)
            lstDateTimes.append(startTime.strftime('%Y-%m-%d|%H:%M:%S'))
    elif iChoice == "Year":
        while startTime < endDateTime:
            startTime += timedelta(year=iValue)
            lstDateTimes.append(startTime.strftime('%Y-%m-%d|%H:%M:%S'))
    return lstDateTimes
def dateTimetoTxt(x):
    if int(x[11:13]) >= 12:
        val = str(int(x[11:13])-12) + ":" + str(x[14:16]) + " PM"
    if int(x[11:13]) < 12:
        val = str(int(x[11:13])) + ":" + str(x[14:16]) + " AM"
    return x[5:7] + "/" + x[8:10] + "/" + x[2:4] + " " + val
allValues = findEveryDateTime(startDate, datetime.time(datetime.datetime.now().hour, datetime.datetime.now().minute),
                              endDate, datetime.time(datetime.datetime.now().hour, datetime.datetime.now().minute),
                              intervalChoice, intervalValue)
allValuesFormatted= []
for i in allValues:
    allValuesFormatted.append(dateTimetoTxt(i))
#st.write(len(allValuesFormatted))
if len(allValuesFormatted) > 100:
    interval.error("Too many points to run")
elif len(allValuesFormatted) > 30:
    interval.warning("Might take too long to run cause too many data points")
interval.markdown("-"*48 + " Calculating **" + str(len(allValuesFormatted)) + "** positions " +"-"*50)
#-------------------------------------------------------------------------------------------

address = st.beta_expander("Address")

addyCol1, addyCol2 = address.beta_columns(2)
addyCol1.subheader("Image Location")
imgSource = addyCol1.radio("",('Latitude and Longitude', 'Address'))
if imgSource == "Latitude and Longitude":
    lat = 34.0958967
    long = -84.259658
    lat = addyCol2.number_input("Latitude",min_value=-90.0000,max_value=90.0000,value=lat,step=0.00001)
    long = addyCol2.number_input("Longitude",min_value=-180.0000,max_value=180.0000,value=long,step=0.00001)
elif imgSource == "Address":
    lat = 0
    long = 0
    addressIN = addyCol2.text_input("Address",max_chars=32)
    addyToLatLongButton = addyCol2.button("Convert to lat long")
    if addyToLatLongButton:
        if not addressIN == "":
            address.error("Please input an address")
        else:
            lat, long = addressToLatLong(addressIN)
#-------------------------------------------------------------------------------------------
st.header("Calculated Data")
address = st.beta_expander("Get Example Image")

#st.header("Get Example Image")
#placeholder = st.empty()
if address.checkbox("Show Location Map"):
    showMap(lat, long)

callImg = address.checkbox("Get Satillite Image")
if callImg:
    Showprogress = st.progress(0.0)
    zoom = 80
    url = getLink(lat, long, zoom)
    Showprogress.progress(0.2)
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    Showprogress.progress(0.25)
    op.add_argument("--start-maximized")
    Showprogress.progress(0.3)
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)
    Showprogress.progress(0.4)
    driver.get(url)
    Showprogress.progress(0.5)
    time.sleep(1)
    Showprogress.progress(0.6)
    closedSideBar = False
    while (not closedSideBar):
        try:
            driver.find_element_by_xpath(
                "/html/body/jsl/div[3]/div[9]/div[3]/div[3]/div/div[2]/div/div[1]/div/button").click()
            closedSideBar = True
        except:
            time.sleep(0.7)
            closedSideBar = False
    Showprogress.progress(0.7)
    takePic = driver.find_element_by_tag_name('body')
    Showprogress.progress(0.8)
    img_b64 = takePic.screenshot_as_base64
    Showprogress.progress(0.9)
    driver.quit()
    Showprogress.progress(1.0)
    #st.write(img_b64[0:10])
    st.image(base64.b64decode(img_b64))
#-------------------------------------------------------------------------------------------
tablePoints = st.beta_expander("Table Points")
tableCol1, tableDataCol2 = tablePoints.beta_columns(2)
with tablePoints:
    tableCol1.subheader("")
    try:
        lenShowTable = tableCol1.number_input("How many points to show out of " + str(len(allValuesFormatted)),
                                              min_value=0,max_value=len(allValuesFormatted),value=10,step=1)
    except:
        lenShowTable = tableCol1.number_input("How many points to show", min_value=0, max_value=len(allValuesFormatted),
                                              value=int(len(allValuesFormatted)), step=1)
    df = pd.DataFrame(allValuesFormatted[0:lenShowTable], columns=["DateTime"])
    tableDataCol2.table(df)
    tableCol1.subheader("")
    specficValue = tableCol1.checkbox("Show Specific Point")
    if specficValue:
        numPT = tableCol1.number_input("Which Point", min_value=0, max_value=len(allValuesFormatted), value=1, step=1)
        tableCol1.write("Point " + str(numPT) + " - " + str(allValuesFormatted[numPT-1]))

#st.header("Address")
#-------------------------------------------------------------------------------------------

st.header('Summary')
st.write("Interval is every", intervalValue, intervalChoice.lower(), " with ", (endDate - startDate).days,
         'days in between \n from', finalDateTimeStart, ' to ', finalDateTimeEnd, " resulting in ", len(allValuesFormatted) ,
         " times at latitude:" , lat ,  "  and longitude:" , long)
#st.markdown('From **' + finalDateTimeStart + '** to **' + finalDateTimeEnd + '**')
finalLatLong = str(lat) + "," + str(long)
#----------------------------------------------------------------------------------------------------------------

def arrowedLine(im, ptA, ptB, width=1, color=(0,255,0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    im = im.copy()
    draw = ImageDraw.Draw(im)
    draw.line((ptA,ptB), width=width, fill=color)
    x0, y0 = ptA
    x1, y1 = ptB
    xb = 0.8*(x1-x0)+x0
    yb = 0.8*(y1-y0)+y0

    if x0==x1:
        vtx0 = (xb-20, yb)
        vtx1 = (xb+20, yb)
    elif y0==y1:
        vtx0 = (xb, yb+5)
        vtx1 = (xb, yb-5)
    else:
        alpha = math.atan2(y1-y0,x1-x0)-90*math.pi/180
        a = 20*math.cos(alpha)
        b = 20*math.sin(alpha)
        vtx0 = (xb+a, yb+b)
        vtx1 = (xb-a, yb-b)
    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im
def drawImg(imgPic, pixelX, pixelY): #Add of different x and y changes
    changeX = pixelX
    changeY = pixelY
    return arrowedLine(imgPic, (2260, 1175),  (2260+changeX, 1175-changeY), width=15, color=(255,0,0))

#-------------------------------------------------------------------------------------------
st.header("Adjust Location")
st.markdown('This is an automated process that uses a control of the mouse on your screen to calculate all the data. ' +
            'Please **DON\'T** touch the mouse during the process')

openApplication = st.button("Get Sample Image")
runDone = False
if openApplication:
    # Open Applicaation
    appApplicationsStartMenu = pygetwindow.getAllTitles()
    if 'Google Earth Pro' not in appApplicationsStartMenu:
        ms.moveTo(522, 2115)
        ms.click()
        ms.typewrite("Google Earth")
        time.sleep(1)
        ms.typewrite(["enter"])
        time.sleep(3)
    else:
        #print("Already Up")
        # Open Google Earth at from task bar
        ms.moveTo(2197, 2104, duration=0.5)
        time.sleep(0.5)
        ms.click()
        # time.sleep(2)
    # Move Window
    x2, y2 = ms.size()
    windows = pygetwindow.getWindowsWithTitle('Google Earth Pro')
    if len(windows) == 0:
        st.stop()
    else:
        window = windows[0]
    window.resizeTo(x2-10, y2-10)
    window.moveTo(0, 0)
    time.sleep(3)
    window.activate()
    time.sleep(1)
    # Close Possible Connection Pop-up
    ms.moveTo(1079, 341, duration=1)
    ms.click()
    time.sleep(1)
    ms.typewrite(["enter"])
    time.sleep(1)
    # Open Terrain Layer
    ms.moveTo(86, 879, duration=1)
    ms.click()
    ms.moveTo(104, 1325, duration=1)
    ms.click()

    # Type In Address
    ms.moveTo(133, 235, duration=1)
    ms.click()
    ms.hotkey("ctrlleft", "a")
    ms.typewrite(finalLatLong)
    ms.typewrite(["enter"])
    time.sleep(10)

    imgExtracted = ms.screenshot()
    st.image(imgExtracted)
    runDone = True
locationCorrect = st.radio("Is the Marker on the center of the parking lot", ("Yes", "No"))
if locationCorrect == "No":
    x2=3840
    y2=2160
    st.write("asdff")
    st.write("Size is", x2, " pixels (X) and ", y2, " pixels (Y)")
    changeValueCol1, changeValueCol2 = st.beta_columns(2)
    changeValueCol1.subheader("Location")
    changeValueCol2.subheader("Zoom")
    lengthChangeX = changeValueCol1.number_input("By how many pixels do you want to marker to move left and right?",
                                                 min_value=-int(x2 / 2), max_value=int(x2 / 2), value=0, step=30)
    lengthChangeY = changeValueCol1.number_input("By how many pixels do you want to marker to move up and down?",
                                                 min_value=-int(y2 / 2), max_value=int(y2 / 2), value=0, step=30)

    zoomChange = changeValueCol2.radio("Zoom Change", ("Yes", "No"))
    if zoomChange == "Yes":
        zoom = changeValueCol2.number_input("How Much Zoom", min_value=0, max_value=25, value=9)
    else:
        zoom = 0
    st.write("-"*70)
    if lengthChangeX == 0 and lengthChangeY == 0:
        st.write("No Location Change")
    elif lengthChangeX > 0 and lengthChangeY == 0:
        st.write("Moving East by ", lengthChangeX)
    elif lengthChangeX < 0 and lengthChangeY == 0:
        st.write("Moving West by ", -lengthChangeX)
    elif lengthChangeX == 0 and lengthChangeY > 0:
        st.write("Moving North by ", lengthChangeY)
    elif lengthChangeX == 0 and lengthChangeY < 0:
        st.write("Moving South by ", -lengthChangeY)
    elif lengthChangeX > 0 and lengthChangeY > 0:
        st.write("Moving North East by ", lengthChangeX, " East and ", lengthChangeY, " North")
    elif lengthChangeX > 0 and lengthChangeY < 0:
        st.write("Moving South East by ", lengthChangeX, " East and ", -lengthChangeY, " South")
    elif lengthChangeX < 0 and lengthChangeY > 0:
        st.write("Moving North West by ", -lengthChangeX, " West and ", lengthChangeY, " North")
    elif lengthChangeX < 0 and lengthChangeY < 0:
        st.write("Moving South West by ", -lengthChangeX, " West and ", -lengthChangeY, " South")
    else:
        st.warning("Error, Check")
    if zoom == 0:
        st.write("No Zoom Change")
    elif zoom > 0:
        st.write("Zooming in by ",zoom)
    elif zoom < 0:
        st.write("Zooming out by ", zoom)
    imgExtracted = ms.screenshot()
    st.image(drawImg(imgExtracted, lengthChangeX, lengthChangeY))












if runDone:
    st.header("Adjust")
    locationCorrect = st.radio("Is the Marker on the center of the parking lot", ("Yes", "No"))
    if locationCorrect == "No":
        #x2, y2 = ms.size()
        #imgExtracted = ms.screenshot()
        #st.image(imgExtracted)

        #"""
        st.write("Size is", x2, " pixels (X) and ", y2, " pixels (Y)")
        changeValueCol1, changeValueCol2 = st.beta_columns(2)
        lengthChangeX = changeValueCol1.number_input("By how many pixels do you want to marker to move left and right?",
                                                     min_value=-int(x2 / 2), max_value=int(x2 / 2), value=0, step=30)
        lengthChangeY = changeValueCol2.number_input("By how many pixels do you want to marker to move up and down?",
                                                     min_value=-int(y2 / 2), max_value=int(y2 / 2), value=0, step=30)

        changeZoomCol1, changeZoomCol2 = st.beta_columns(2)
        zoomChange = changeZoomCol1.radio("Zoom Change", ("Yes", "No"))
        if zoomChange == "Yes":
            zoom = changeZoomCol2.number_input("How Much Zoom", min_value=0, max_value=25, value=9)
        else:
            zoom = 0

        if lengthChangeX == 0 and lengthChangeY == 0:
            st.write("No Location Change")
        elif lengthChangeX > 0 and lengthChangeY == 0:
            st.write("Moving East by ", lengthChangeX)
        elif lengthChangeX < 0 and lengthChangeY == 0:
            st.write("Moving West by ", -lengthChangeX)
        elif lengthChangeX == 0 and lengthChangeY > 0:
            st.write("Moving North by ", lengthChangeY)
        elif lengthChangeX == 0 and lengthChangeY < 0:
            st.write("Moving South by ", -lengthChangeY)
        elif lengthChangeX > 0 and lengthChangeY > 0:
            st.write("Moving North East by ", lengthChangeX, " east and ", lengthChangeY, " north")
        elif lengthChangeX > 0 and lengthChangeY < 0:
            st.write("Moving South East by ", lengthChangeX, " east and ", -lengthChangeY, " south")
        elif lengthChangeX < 0 and lengthChangeY > 0:
            st.write("Moving North West by ", lengthChangeX, " west and ", lengthChangeY, " north")
        elif lengthChangeX < 0 and lengthChangeY < 0:
            st.write("Moving South West by ", lengthChangeX, " west and ", -lengthChangeY, " south")
        else:
            st.stop()
        if zoomChange == "Yes":
            st.write("Zooming in by ", zoom, "notches")
        else:
            st.write("No zoom change ", zoom, "notches")

        st.image(drawImg(imgExtracted, lengthChangeX, lengthChangeY))


        runNewCalc = st.button("Run Changes")

        if runNewCalc:
            ms.moveTo(int(x2 / 2), int(y2 / 2), duration=0.5)
            ms.mouseDown(button='left')
            ms.moveRel(lengthChangeX, lengthChangeY, duration=1)
            ms.mouseUp(button='left')

            for i in range(zoom):
                ms.scroll(1)
                time.sleep(0.1)
            ms.typewrite(["n", "u", "r"])
            time.sleep(2)
            imgExtractedNew = ms.screenshot()
            st.write("New Image")
            st.image(imgExtractedNew)
        #"""
st.header('Run Calculations')
st.write('This process uses a control of the mouse on your screen to calculate all the data. Please don\'t touch the mouse during the process')


runCalculations = st.button("Run " + str(len(allValuesFormatted)) + " images")
imgLST = []
if runCalculations:
    # Close Sidebar
    ms.moveTo(775, 158, duration=0.5)
    ms.click()
    # Open Historical Imagery
    ms.moveTo(212, 81, duration=1)
    ms.click()
    ms.moveTo(220, 797, duration=1)
    ms.click()
    for i in allValuesFormatted:
        # Import More Detail Box
        ms.moveTo(798, 238, duration=0.3)
        ms.click()

        # Type In Own Data
        time.sleep(2)
        ms.moveTo(1889, 790)
        ms.click()
        ms.hotkey("ctrlleft", "a")
        ms.typewrite(i)

        # Click Okay
        ms.moveTo(2412, 1420)
        ms.click()


        #findGreen = ms.locateOnScreen('greenSquare.png', confidence=1.0)
        #if not findGreen:
        #    st.write("Found Green Screen, waiting")
        #ctRun = 0
        #while findGreen == None:
        #    findGreen = ms.locateOnScreen('greenSquare.png', confidence=1.0)
        #    time.sleep(1)
        #    ctRun += 1
        #    if ctRun >= 10:
        #        findGreen = "BREAK"
        time.sleep(2)
        pic = ms.screenshot()
        imgLST.append(pic)
    #, colImg4, colImg5, colImg6, colImg7, colImg8, colImg9, colImg10
    colImg1, colImg2, colImg3 = st.beta_columns(3)
    colImg1.header("Img 1")
    colImg2.header("Img 2")
    colImg3.header("Img 3")
    ct = 1
    for i in imgLST:
        if ct % 3 == 0:
            colImg1.image(i)
        elif ct % 2 == 0:
            colImg2.image(i)
        else:
            colImg3.image(i)
        ct += 1
"""
testDataSet = [str(1),str(3),str(4),str(6)]

#list_string = map(str, list_int)
def convertLST_Int_2_Str(lst):
    return list(map(str, lst))


df = pd.DataFrame({
  'Parking Lot': testDataSet,
  'Dates': [10, 20, 30, 40]
})

df = df.rename(columns={'Parking Lot':'index'}).set_index('index')
st.write("ad")
st.write(df)
st.line_chart(df)
"""


st.header("Other Process")
learnMore = st.beta_expander("Learn More")

learnMore.title('Understand the trends of a parking lot.')
#learnMore.markdown("## RUN")
#st.header('')
learnMore.markdown("Run calculations on how your foot-traffic is doing. Understand the trends of your bussiness.")
#st.write("")
learnMore.markdown("We are able to calculate the number of cars in any specific parking lot and understand the foot "+
                   "traffic of any location over the course of multiple months.")
learnMore.subheader("How it works")
learnMore.markdown("A user inputs its duration from when to when they want their data extracted, and using **Google Earth Engine**" +
                   "the program automatically extracts satilliet data directly from its location. Using these different images, " +
                   "the program runs an AI script to detect the number of cars in the parking lot. Following, it runs different statistical " +
                   "calculations understanding how the different number of cars are changing over the year.")
learnMore.header('Learn More')
learnMore.header('Do it now!')

calcOther = st.sidebar.beta_expander("Other Testing")

rh_l = calcOther.slider('R/H L', 0, 255, 0)
gs_l = calcOther.slider('G/S L', 0, 255, 0)
bv_l = calcOther.slider('B/V L', 0, 255, 0)

rh_h = calcOther.slider('R/H H', 0, 255, 255)
gs_h = calcOther.slider('G/S H', 0, 255, 255)
bv_h = calcOther.slider('B/V H', 0, 255, 255)

calcOtherMain = st.beta_expander("Display Other Calculations")
lower = np.array([0,0,0], dtype = "uint8")
upper = np.array([255,255,255], dtype = "uint8")

image_TESTING= "/Users/kunal/Desktop/A.png"
img_init = cv2.imread(image_TESTING)
img_hsv= cv2.cvtColor(img_init, cv2.COLOR_BGR2HSV)

lower = np.array([rh_l, gs_l, bv_l], dtype="uint8")
upper = np.array([rh_h, gs_h, bv_h], dtype="uint8")

img_out = img_hsv.copy()
# Find the pixels that correspond to road
img_out2 = cv2.inRange(img_out,lower, upper)
# Clean from noisy pixels and keep only the largest connected segment
img_out = post_process(img_out2)

col1, col2 = calcOtherMain.beta_columns(2)
col1.header("Original")
col1.image(img_init, use_column_width=True)

col2.header("IMG HSV")
col2.image(img_hsv, use_column_width=True)

if rh_l == 0 and gs_l == 0 and bv_l == 0 and rh_h == 255 and gs_h == 255 and bv_h == 255:
    calcOtherMain.write("Use Side bar to control image results below")
col3, col4 = calcOtherMain.beta_columns(2)
col3.header("Controlled Img 1")
col3.image(img_out2, use_column_width=True)

col4.header("Controlled Img 2")
col4.image(img_out, use_column_width=True)




st.header("----"*18)


