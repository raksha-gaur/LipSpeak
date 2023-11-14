#Importing dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import moviepy.editor as moviepy 
#Set layout wide
st.set_page_config(layout='wide')

#Basic sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('lipSpeak')
    st.info('Thanks for using me. I am at development stage so please forgive if any mistakes!')

st.title('LipSpeak App') 
#Generating a list of options
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

#Making two options
col1, col2 = st.columns(2)

if options:
    with col1:
        file_path = os.path.join('..', 'data', 's1', selected_video)
        if os.path.isfile(file_path):
        # Run FFmpeg command to convert the video
            clip = moviepy.VideoFileClip(file_path)
            clip.write_videofile("test_video.mp4")
        # Check if the converted 'test_video.mp4' exists
            if os.path.exists('test_video.mp4'):
                with st.container():
                    st.info('The video below displays the converted video in mp4 format')
                    video = open('test_video.mp4', 'rb')
                    video_bytes = video.read()
                    st.video(video_bytes)
            else:
                st.error("Conversion to 'test_video.mp4' failed.")
        else:
            st.error(f"File not found: {file_path}")


    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 
        

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)