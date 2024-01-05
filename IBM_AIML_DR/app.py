import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model("Retinopathy.h5")
@app.route('/')
def home():
    return render_template("main.html")


@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/logout')
def logout():
    return render_template("success.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(120,120))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        pred=np.argmax(model.predict(x),axis=1)
        #index = ['0', '1', '2', '3', '4']
        index = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']
        text="The image is detected as : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()