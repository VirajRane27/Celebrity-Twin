# import os
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import InceptionResNetV2
# from keras_vggface.vggface import VGGFace
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from facenet_pytorch import InceptionResnetV1, MTCNN
# import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import cv2
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import streamlit as st


detect = MTCNN()
celeb = os.listdir('Celeb Faces')

# print(celeb)

path = []

for i in celeb:
    for j in os.listdir(os.path.join('Celeb Faces', i)):
        path.append(os.path.join('Celeb Faces', i, j))

# print(len(path))

# model = InceptionResnetV1(pretrained='vggface2').eval()
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model = VGGFace(model='vggface', include_top=False, input_shape=(224, 224, 3))

# features = []
# def feature_path(path, model):
#     img = image.load_img(path, target_size=(224, 224))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     features = model.predict(img).flatten()
#     return features

# features3 = []
def feature_img(img, model):
  img = np.array(img)
  img = cv2.resize(img, (224, 224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)
  features = model.predict(img).flatten()
  return features

# def feature_img(img, model):
#     img = mtcnn(img)
#     if img is not None:
#         with torch.no_grad():
#             res = model(img).cpu().numpy()
#             return res

def valid(img):
    try:
        face = detect.detect_faces(np.array(img))
        confi = face[0]['confidence']
        if face and confi > 0.9:
            return 1
        else:
            return -1
    except Exception as e:
        st.error("The uploaded image is not valid for face detection. Please try again with a clear image.")
        return -2

def crop(img):
    faces = detect.detect_faces(np.array(img))
    if faces:
        x, y, width, height = faces[0]['box']
        img = np.array(img)
        img = img[y:y+height, x:x+width]
        return img
    return None

def recommend(feature, a):
    similarty = []
    for i in range(len(a)):
     similarty.append(cosine_similarity(feature.reshape(1,-1), a[i].reshape(1,-1))[0][0])
    index = sorted(list(enumerate(similarty)), reverse=True, key=lambda x:x[1])[0][0]
    return index

# features3 = []
# for i in range(len(path)):
#   img = Image.open(path[i])
#   features3.append(feature_img(img, model))
#   print(len(features3))
# pickle.dump(features3, open('features3.pkl', 'wb'))

a = pickle.load(open('features3.pkl', 'rb'))


st.title('Which Celebrity are You ?')

your_image = st.file_uploader('Upload your Image', type=['jpg', 'jpeg', 'png'])

if your_image is not None:
    img = Image.open(your_image)
    if valid(img) == 1:
        img = crop(img)
        if img is not None:
            feature = feature_img(img, model)
            index = recommend(feature, a)

            col1, col2 = st.columns(2)

            with col1:
                st.header('Your Uploaded Photo')
                your_image = Image.open(your_image)
                your_image = your_image.resize((250, 300))
                st.image(your_image)

            with col2:
                name = " ".join(path[index].split('\\')[1].split('_'))
                st.header('You look like ' + name)
                celeb = Image.open(path[index])
                celeb = celeb.resize((250, 300))
                st.image(celeb)
        else:
            st.header('Image is not clear or face is not recognized')
            st.header('Try with other Image')
    else:
        st.header('Image is not clear or face is not recognized')
        st.header('Try with other Image')