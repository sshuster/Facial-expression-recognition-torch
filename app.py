import streamlit as st
from PIL import Image
import cv2
import torch
from torchvision import transforms
from vgg import VGG
from datasets import FER2013
from utils import eval, detail_eval
from face_detect.haarcascade import haarcascade_detect
import numpy as np

def detect(model, image):
    crop_size = 44
    transform_test = transforms.Compose([
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

    original_image = np.array(image)
    original_image = original_image[:, :, ::-1].copy() 
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    faces = haarcascade_detect.face_detect(gray_image)
    if faces != []:
        for (x, y, w, h) in faces:
            roi = original_image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            roi_gray = Image.fromarray(np.uint8(roi_gray))
            inputs = transform_test(roi_gray)
            
            ncrops, c, ht, wt = np.shape(inputs)
            inputs = inputs.view(-1, c, ht, wt)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(ncrops, -1).mean(0)
            _, predicted = torch.max(outputs, 0)
            expression = classes[int(predicted.cpu().numpy())]
            
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            text = "{}".format(expression)
            
            cv2.putText(original_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
    return original_image
def run():
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    crop_size= 44
    trained_model = torch.load("C:/Users/Admin/Downloads/model_state.pth.tar")
    model = VGG("VGG19")
    model.load_state_dict(trained_model["model_weights"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    st.title("Facial expression recognition")
    img_file = st.file_uploader("Upload an image", type= ["png", "jpg", "jpeg"])

    if img_file is None:
        st.write('** Please upload an image **')
    original_image = Image.open(img_file, mode='r')
    st.image(original_image, use_column_width= True)
    model = 1
    if st.button('Predict'):
        predict_image = detect(model, original_image)
        image = Image.fromarray(cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB))
        st.image(image, use_column_width= True)







if __name__ == "__main__":
    run()