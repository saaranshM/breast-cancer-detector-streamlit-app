# importing importatnt libraries
import streamlit as st
from PIL import Image, ImageOps
import os
import requests
import cv2
import shutil
import pandas as pd
import base64

# start of streanlit app
st.set_page_config(page_title="Breast Cancer Detector")
st.title("Breast Cancer Detector")

# explanation of app
st.header("App to detect Invasive ductal carcinoma(IDC) from cancer tissue images.")
st.subheader("This app uses a deep learning model which is deployed as an API, that can be used to ease the work of the pathologist so that they can check the tissue samples in greater batches and in a more efficient manner.")

# upload images function
uploaded_files = st.file_uploader(
    "Choose samples to upload...", accept_multiple_files=True)


display_image_list = []  # list of images to display
files = []  # list of images to send

# path for images to be saved
UPLOAD_FOLDER = os.getcwd()
UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'saved_images')


# function to add white border to images while displaying
def add_border(input_image, border, color=0):
    """
    Accepts PIL Image and returns an image with an image with "border" number of
    pixels around it.
    in: imput_image(PIL Image), border(int), color(str)
    out: bimg(PIL Image)
    """
    img = Image.open(input_image)
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
        return bimg
    else:
        raise RuntimeError('Border is not an integer or tuple!')


def save_image(input_image):  # function to save images
    """
    Function to save Image to directory after uploading
    in: ImgFile(bytes data)
    out: Image saved Location(str)
    """
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    IMG_NAME = input_image.name
    save_img = Image.open(input_image)
    img_save_location = os.path.join(UPLOAD_FOLDER, IMG_NAME)
    save_img.save(img_save_location)
    return img_save_location, IMG_NAME


#
count = 0
# saves images to upload and display to lists
for n, uploaded_file in enumerate(uploaded_files):
    count += 1
    image = add_border(uploaded_file, border=10, color='white')
    saved_img_location, saved_image_name = save_image(uploaded_file)
    files.append((
        'images', (
            saved_image_name, open(saved_img_location, 'rb'),
            'image/png'
        )
    ))
    display_image_list.append(image)


def predict():  # function that sends request to api
    url = 'https://breastcancerapi.herokuapp.com/'
    response = requests.post(url, files=files)
    return response.json()


# function that generates to link to download csv file of predictions
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="breast_cancer_predictions.csv">Download predictions as csv file</a>'
    return href


if count >= 1:
    st.subheader("Uploaded Images :")
    st.image(display_image_list, width=150)

    if st.button("Predict Results"):
        with st.spinner("Fetching Predictions...."):
            predictions_dict = predict()
        model_predictions = predictions_dict['model_predictions']

        df = pd.DataFrame(model_predictions)
        df["prediction"].replace(
            {1: "detected", 0: "no detection"}, inplace=True)

        st.subheader("Model Predictions:")
        st.write(df)

        href = get_table_download_link(df)
        st.markdown(href, unsafe_allow_html=True)
