import streamlit as st
from streamlit_folium import folium_static
import folium

"# streamlit-folium"

with st.echo():
    import streamlit as st
    from streamlit_folium import folium_static
    import folium

    # center on Liberty Bell
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)

    # add marker for Liberty Bell
    tooltip = "dasfewfa Bell"
    folium.Marker(
        [39.949610, -75.150282], popup="Liberty Bell", tooltip=tooltip
    ).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)

#IMPORT ALL NECCESSARY LIABRIES
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image, ImageDraw
import shutil
import time
import pytesseract

threshold = 80


#----------------------------------------------------------------------------------------
#TITLES
st.title('Drive Model Detection Test')
#RANDOM TEST THING TO SEE IF IT IS RUNNING
x = st.button('CLICK BUTTON TO SEE IF IT IS RUNNING?')
if x:
    st.balloons()
    st.write("test")

@st.cache
def get_data():
    url = '/root/Objectdetection/Data/Drive.csv'
    return pd.read_csv(url)
def get_list(path):
    fileslist = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            fileslist.append(file)
    return fileslist

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
def FrameCapture(path, buffer):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()
        try:
            cv2.imwrite("frame%d.jpg" % count, image)
        except:
            pass
            st.exception("Either loop has reached last frame or one frame is errored out")
        count += buffer
    st.success("Finished with "+ count+ " images")

def getTotalFrame(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success = vidObj.read()
        count+=1
    return count
def extract_image_one_fps(video_source_path, numsec):
    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * numsec))  # 2 second***
        success, image = vidcap.read()
        try:
            cv2.imwrite("frame%d.png" % count, image)  # save frame as PNG file
        except:
            break
        if showoptions:
            st.write('{}.sec reading a new frame:{}'.format(count, success))
        count += 1
    st.success("Finished with " + str(count) + " images")

directory = st.text_input('Type the file path that goes into your folder containing the video file', '/root/ObjectdetectionData/')

showoptions = st.sidebar.checkbox("Show more options?")

try:
    os.chdir(directory)
    st.success("WORKING")
except:
    st.warning("NOT WORKING")

st.sidebar.markdown("The current working directory is")
st.sidebar.markdown(os.getcwd())


model_path_way = directory
json_path_way = directory + "json/model_class.json"

model_path_final = model_path_way

model_files = []
for i in os.listdir(model_path_final):
    if i.endswith(".h5"):
        model_files.append(i)

option = st.selectbox( 'Which model do you want to use?', (model_files))

model_path_final += "/"+option
if showoptions:
    json_path_final = st.text_input("Upload the Json File", json_path_way)
else:
    json_path_final = json_path_way
st.markdown("-----------------------------------")
st.title("Final Folders")
st.write("The Model file is ")
st.success(model_path_final)
st.write("The JSON file is ")
st.success(json_path_final)


st.markdown("-----------------------------------")
st.title("Upload Testing Video")
videopath = st.text_input("Upload the testing video folder", '/Users/kunal/Documents/VdartCode/streamlit/testingfiles/RunningVideo')
st.write(videopath)
videos = []

for i in os.listdir(videopath):
    if i.endswith(".mp4"):
        videos.append(i)

optionVid = st.selectbox( 'Which video do you want to use?', (videos))

video_filepath = videopath+"/"+optionVid


dire = st.checkbox("Move all other videos to extra folder")


if dire:
    os.chdir(videopath)
    runningVideo = videopath + "/RunningVideoNot"
    if not os.path.exists(runningVideo):
        os.makedirs("RunningVideoNot")
    for i in os.listdir(videopath):
        if not i == (optionVid):
            shutil.move(i, runningVideo)
    st.write("Moved")

st.write("The video path is ")
st.success(video_filepath)


st.markdown("-----------------------------------")
st.title("Running the Testing Video")

st.header("Spliting the video in multiple frames")
buff = st.slider('After how many frames each frame will be taken?', 1, 10000, 1000)


shouldUserSplit = False
count = 0
for file in os.listdir(videopath):
    if file.startswith("frame"):
        count+=1
if count != 0:
    st.write("The frames are already there, dont split it")
    shouldUserSplit = False
else:
    st.write("Please Split the frame")
    shouldUserSplit = True



split = st.button("Split it?")

if split:
    if shouldUserSplit == True:
        os.chdir(videopath)
        extract_image_one_fps(video_filepath, buff)
    elif shouldUserSplit == False:
        st.warning("The frames are already there, don't split it")
count = 0
for file in os.listdir(videopath):
    if file.startswith("frame"):
        count+=1
if count != 0:
    st.write("The frames are already there, don't split it")
    shouldUserSplit = False



fileslist = get_list(videopath)

st.write("There are ", len(fileslist), " images in the array")


finalfiles = []
finalpicnames = []
for i in fileslist:

    finalfiles.append(videopath+"/"+i)
    finalpicnames.append(i)

#----------------------------------------------------------------

st.markdown("-----------------------------------")
st.title("Import")

importt = st.checkbox("Import Base?")
number_of_classes = st.slider('How many different classes do you have?', 1, 10, 2)
if importt:
    from imageai.Prediction.Custom import ModelTraining
    from imageai.Prediction.Custom import CustomImagePrediction

    st.write("DONE IMPORTING")

runFinal = st.checkbox("Model Import?")
if runFinal:

    prediction = CustomImagePrediction()

    prediction.setModelTypeAsResNet()
    prediction.setModelPath(model_path_final)
    prediction.setJsonPath(json_path_final)
    prediction.loadModel(num_objects=number_of_classes)


run = st.radio("Running:", ('Select','Run One Picture', 'Run Entire Video'))
if run == 'Select':
    st.markdown("Please select what to do")
if run == 'Run One Picture':
    singlePicTest = st.selectbox( 'Test Specific Picture?', (finalpicnames))
    singlePicFilePath = videopath+"/"+singlePicTest
    st.write("The Photo is " + singlePicTest)
    image = Image.open(singlePicFilePath)
    st.image(image, caption="This is the testing image", use_column_width=True)


    st.markdown("-----------------------------------")
    st.title("Run")
    runFinalpart2 = st.checkbox("Run the Final using model " + option + "?")
    predictions = 0
    probabilities = 0
    runResults = False
    if runFinalpart2:
        predictions, probabilities = prediction.predictImage(singlePicFilePath, result_count=2)
        st.write(predictions)
        st.write(probabilities)
        runResults = True
    runOCR = False
    if predictions is not 0:
        if showoptions:
            st.write(predictions)

        for i in range(len(predictions)):
            str = "There is a "+ predictions[i]+ "with a "+ probabilities[i] + " chance Probabilty"
            st.write(str)

            prob = round(float(probabilities[0]))
            if(prob > 80):
                runOCR = True


    st.markdown("-----------------------------------")
    st.title("OCR")
    if runOCR:
        OCRPIC = singlePicFilePath
        value=Image.open(OCRPIC)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        text = pytesseract.image_to_string(value, config='')
        st.write("The text detected is")
        st.write(text)

    st.markdown("-----------------------------------")
    st.title("Results")
    if runResults == True:
        if predicitions is not 0:
            prob = round(float(probabilities[0]))
            if prob > threshold:
                st.write("The image ", singlePicTest, " has a head unit", "with a ", prob, "probability.")

os.chdir(videopath)
if run == "Run Entire Video":
    results_array = prediction.predictMultipleImages(finalpicnames, result_count_per_image=2)
    dfyy = pd.DataFrame(results_array, columns=['HeadUnit', 'Probability', 'Frame'])
    save_results = []
    count = -1
    for i in results_array:
        headunit, prob = i["predictions"], i["percentage_probabilities"]

        for idx in range(len(headunit)):
            # print(pred[idx] , " : " , prob[idx])
            if (headunit[idx] == "HeadUnit"):
                count += 1
            save_results.append((headunit[idx], prob[idx], count))

    df = pd.DataFrame(save_results, columns=['HeadUnit', 'Probability','Frame'])
    st.write("Working with ", len(df)/2, " frames")

    showText = st.checkbox("Show All the text?")
    containHeadUnit = []
    count = 0
    for i in df["HeadUnit"]:
        if i == "HeadUnit":
            runningFrame = float(df.iloc[count]["Probability"])
            if runningFrame > threshold:
                path = (df.iloc[count]["Frame"])
                pic = "frame"+str(path)+".png"
                if showText:
                    st.write(pic)
                value2 = Image.open(pic)
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                text = pytesseract.image_to_string(value2, config='')
                if showText:
                    st.markdown(text)
                df['OCR'] = text

        count +=1
    st.write(df)
    createspread = st.checkbox("Create a spreadsheet")
    if showoptions:
        dataDir = st.text_input("Input the directory where you want to save the EXCEL Spreadsheet",
                                (directory + "FinalData"))
        if createspread:
            file_ext = str(random.randint(1, 100))
            os.chdir(dataDir)
            df.to_csv(os.path.join(execution_path, "save_results" + file_ext + ".csv"), index=False, encoding='utf8')
            st.write("Saved the excel to ")
            st.success(execution_path, "save_results" + file_ext + ".csv")

showfile = st.sidebar.checkbox("Show files in working directory?")
if showfile:
    for i in os.listdir():
        st.sidebar.checkbox(i)
showgraph = st.sidebar.checkbox("Show that many charts and graphs?")
showspecific = st.sidebar.checkbox("Show specific values for different epochs?")
showspecificGraph = st.sidebar.checkbox("Show specific graphs for different epochs?")


st.sidebar.markdown("Upload the file with the model data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    df = data
else:
    df = get_data()

df = df.drop('Epoch', axis=1)
if showgraph:
    st.header("Table")
    st.write(df)
    st.header("Area Chart")
    st.area_chart(df)
    st.header("Bar Chart")
    st.bar_chart(df)
    st.header("Everyone's Line Chart")
    st.line_chart(df, use_container_width=True)


if showspecific:
    epoch = st.slider('What epoch do you want', 1, 30, 15)
    st.write("Showing data for epoch number ", epoch, ':')

    st.write(df.loc[epoch])

if showspecificGraph:
    st.header("Showing specific Graphs")
    type = st.radio(
         "Which Graph do you want to see",
         ('Accuracy', 'Loss', 'Validation Accuracy', 'Validation Loss'))

    if type == 'Accuracy':
        st.write('You selected Accuracy. Here is the Graph for it')
        dfAcc = df.drop('Validation_loss', axis=1)
        dfAcc = dfAcc.drop('Validation_Acc', axis=1)
        dfAcc = dfAcc.drop('Loss', axis=1)
        st.write(dfAcc)
        st.line_chart(dfAcc)
    elif type == 'Loss':
        dfLoss = df
        st.write('You selected Loss. Here is the Graph for it')
        dfLoss = dfLoss.drop('Accuracy ', axis=1)
        dfLoss = dfLoss.drop('Validation_loss', axis=1)
        dfLoss = dfLoss.drop('Validation_Acc', axis=1)
        st.write(dfLoss)
        st.line_chart(dfLoss)
    elif type == 'Validation Accuracy':
        st.write('You selected Validation Accuracy. Here is the Graph for it')
        dfValAcc = df.drop('Validation_loss', axis=1)
        dfValAcc = dfValAcc.drop('Loss', axis=1)
        dfValAcc = dfValAcc.drop('Accuracy ', axis=1)
        st.write(dfValAcc)
        st.line_chart(dfValAcc)
    elif type == 'Validation Loss':
        st.write('You selected Validation Loss. Here is the Graph for it')
        dfValLos = df.drop('Loss', axis=1)
        dfValLos = dfValLos.drop('Validation_Acc', axis=1)
        dfValLos = dfValLos.drop('Accuracy ', axis=1)
        st.write(dfValLos)
        st.line_chart(dfValLos)
    else:
        st.write("ERROR!!")

