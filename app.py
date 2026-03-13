import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time

# PAGE CONFIG
st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🤖",
    layout="wide"
)

# DARK FUTURISTIC STYLE
st.markdown("""
<style>

body{
background-color:#0f172a;
color:white;
}

.big-title{
font-size:50px;
text-align:center;
color:#38bdf8;
font-weight:bold;
}

.subtitle{
text-align:center;
font-size:20px;
color:#94a3b8;
}

</style>
""",unsafe_allow_html=True)

st.markdown('<p class="big-title">🤖 AI Digit Recognizer</p>',unsafe_allow_html=True)
st.markdown('<p class="subtitle">Draw or Upload a digit and let AI predict</p>',unsafe_allow_html=True)


# LOAD + TRAIN MODEL
@st.cache_resource
def train_model():

    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

    x_train=x_train/255
    x_test=x_test/255

    x_train=x_train.reshape(-1,28,28,1)
    x_test=x_test.reshape(-1,28,28,1)

    model=models.Sequential([

        layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(64,activation='relu'),

        layers.Dense(10,activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history=model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test,y_test),
        verbose=0
    )

    loss,acc=model.evaluate(x_test,y_test,verbose=0)

    return model,history,acc


model,history,accuracy=train_model()

# DASHBOARD
st.sidebar.title("📊 Model Dashboard")
st.sidebar.metric("Test Accuracy",f"{accuracy*100:.2f}%")
st.sidebar.write("Dataset: MNIST")

# LIVE TRAINING GRAPH
st.subheader("📈 Training Graph")

fig,ax=plt.subplots()

ax.plot(history.history['accuracy'],label="Train Accuracy")
ax.plot(history.history['val_accuracy'],label="Validation Accuracy")

ax.legend()

st.pyplot(fig)

st.divider()

# DRAW CANVAS
st.subheader("✏ Draw Digit")

canvas_result = st_canvas(
fill_color="black",
stroke_width=18,
stroke_color="white",
background_color="black",
height=280,
width=280,
drawing_mode="freedraw"
)

# IMAGE UPLOAD
st.subheader("📤 Upload Digit Image")

uploaded_file = st.file_uploader(
"Upload a digit image",
type=["png","jpg","jpeg"]
)

def preprocess(img):

    img=img.convert("L")

    img=img.resize((28,28))

    img=np.array(img)/255.0

    img=img.reshape(1,28,28,1)

    return img


def predict_digit(img):

    prediction=model.predict(img)

    digit=np.argmax(prediction)

    confidence=np.max(prediction)

    return digit,confidence,prediction


# PREDICT BUTTON
if st.button("🚀 Predict Digit"):

    if canvas_result.image_data is not None:

        img=Image.fromarray((canvas_result.image_data[:,:,0]).astype('uint8'))

        img=preprocess(img)

        digit,confidence,pred=predict_digit(img)

    elif uploaded_file:

        img=Image.open(uploaded_file)

        img=preprocess(img)

        digit,confidence,pred=predict_digit(img)

    else:

        st.warning("Draw or Upload a digit first")
        st.stop()

    # ANIMATED METER
    st.subheader("🤖 AI Prediction")

    progress=st.progress(0)

    for i in range(int(confidence*100)):
        time.sleep(0.01)
        progress.progress(i+1)

    st.success(f"Predicted Digit: {digit}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Prediction Probability")

    st.bar_chart(pred[0])