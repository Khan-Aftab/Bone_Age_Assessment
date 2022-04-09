from pyexpat import features
import streamlit as st2
import pandas as pd
import string   
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model


st2.set_page_config(layout="wide")
st2.markdown(
    """
<style>
.main {
background-color: #F5F5F5;
}
</style>
""",
    unsafe_allow_html=True,
)


header=st2.container()
dataset=st2.container()
features=st2.container()    
model=st2.container()


with header:
    st2.markdown("<h1 style='text-align: center; color: black;'>Welcome to our Bone Age Assessment Project </h1>", unsafe_allow_html=True  )
    st2.write('Bone Age Assessment through radiographs of the left hand is widely used in the diagnosis, treatment and monitoring of endocrine, genetic and growth disorders in children. Absent birth data is a big problem in our part of the world. Around 65% of all births are not registered by the age of 5 years. Thus the need for accurate estimation of age arises in conditions where the age of a child needs to be accurate, such as during immigration, in lawsuits and in competitive sports. In these cases bone age is used to provide the closest estimate of chronological age.       Bone age assessment is an essential tool in the evaluation and follow-up of GHD patients. The bone age study can help evaluate how fast or slowly a child''s skeleton is maturing, which can help doctors diagnose conditions that slow down or speed up physical growth and development. This test is usually ordered by pediatricians or pediatric endocrinologists.')

    st2.write('The bone age test can be used to predict the following: ')
    st2.write('1. How much time a child will be growing')
    st2.write('2. When a child will start puberty')
    st2.write('3. What the child’s final height will be')

    st2.write('The test also can help doctors monitor progress and guide treatment of kids with conditions that affect growth, including diseases that affect the levels of growth hormones, such as growth hormone deficiency, hypothyroidism, precocious puberty, and adrenal gland disorders, genetic growth disorders, such as Turner syndrome and orthopedic or orthodontic problems in which the timing and type of treatment (surgery, bracing, etc.) are guided by the child''s expected growth.')
    st2.write(' ')

with dataset:
    st2.header('Dataset- RSNA Bone Age Assessment ')
    st2.markdown("The dataset was provided by the Radiology Society of North America. It consists of 14,000 hand radiograph images and their corresponding bone ages in months. The snapshot of the training and testing data is shown below: ", unsafe_allow_html=True  )

    left_table, right_table = st2.columns(2)

    with left_table:
        st2.text('Training dataset :')
        baa_training_data = pd.read_csv('data/boneage-training-dataset.csv')
        st2.write(baa_training_data.head(10))
   
    with right_table:
        st2.text('Test dataset :')
        baa_test_data = pd.read_csv('data/boneage-test-dataset.csv')
        st2.write(baa_test_data.head(10))


    st2.header('Dataset Visualization: ')
    st2.subheader('Boneage distribution on the RSNA dataset(in months)')
    boneage_dist = pd.DataFrame(baa_training_data['boneage'].value_counts()).head(70)
    st2.bar_chart(boneage_dist)


with model:
    st2.header('Time to run the assessment... ')
    


#st.title('Bone Age Assessment')
std_bone_age = 127.3207517246848
mean_bone_age = 41.18202139939618

model = load_model(r"C:\Users\khana\Downloads\Telegram_Desktop\Major_Project\Major_Project\models\best_model.h5", compile=False)

upload_file = st2.file_uploader(label='Upload the hand radiograph', type=["jpg","png","jpeg"])

def predict(image_data,model):
    image_array = np.array(Image.open(image_data).convert("RGB").resize((256,256)))
    images = image_array/255.0
    image = tf.expand_dims(images, 0)
    predictions = std_bone_age+mean_bone_age*(model.predict(image,batch_size=32))
    predict = predictions/12.0
    predict = predict[0][0]

    return predict

if upload_file is None:
    st2.text("Please Upload the hand radiograph !")
else:
    prediction = predict(upload_file,model) 
    result = f"The bone age is {prediction}  Years"
    st2.success(result)





