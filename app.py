import streamlit as st
from PIL import Image, ImageOps
import os
import requests
import cv2
import json
import shutil
import pandas as pd
import base64

st.title("Breast Cancer Detector")

st.header("App to detect Invasive ductal carcinoma(IDC) from cancer tissue images.")
st.subheader("This app uses a deep learning model which is deployed as an API, that can be used to ease the work of the pathologist so that they can check the tissue samples in greater batches and in a more efficient manner.")

uploaded_files = st.file_uploader(
    "Choose samples to upload...", accept_multiple_files=True)

display_image_list = []
image_captions = []
image_data = {
    'images': []
}
files = []

UPLOAD_FOLDER = os.getcwd()
UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'saved_images')


def add_border(input_image, border, color=0):
    img = Image.open(input_image)
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
        return bimg
    else:
        raise RuntimeError('Border is not an integer or tuple!')


def save_image(input_image):
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    IMG_NAME = input_image.name
    save_img = Image.open(input_image)
    img_save_location = os.path.join(UPLOAD_FOLDER, IMG_NAME)
    save_img.save(img_save_location)
    return img_save_location, IMG_NAME


count = 0
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
    image_captions.append(uploaded_file.name)


def predict():
    url = 'http://127.0.0.1:5000/'
    response = requests.post(url, files=files)
    # shutil.rmtree(UPLOAD_FOLDER)
    return response.json()


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
    st.image(display_image_list, width=120)

    if st.button("Predict Results"):
        predictions_dict = predict()
        model_predictions = predictions_dict['model_predictions']
        print(model_predictions)
        df = pd.DataFrame(model_predictions)
        df["prediction"].replace(
            {1: "detected", 0: "no detection"}, inplace=True)
        st.subheader("Model Predictions:")
        st.write(df)
        href = get_table_download_link(df)
        st.markdown(href, unsafe_allow_html=True)
