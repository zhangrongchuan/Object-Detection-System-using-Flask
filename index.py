from flask import Flask, render_template, request
from models.test_single_image import detect_single_image as detect
import models.config as config
from models.decode import Decode
from model import DarkNet
from PIL import Image
import torch

app = Flask(__name__)

@app.route("/" , methods=["GET","POST"])

def index():
    return render_template("index.html")

@app.route("/result",methods=["GET","POST"])

def get_result():
    image=Image.open(request.files.get("image"))
    image.save("static/pic/image.png")
    model=DarkNet()
    model.load_state_dict(torch.load(config.WEIGHT_RESTORE_PATH))
    model.train()
    model.eval()
    decoder=Decode()
    location_result=detect(image,model,decoder).tolist()
    print(location_result)
    return render_template("result.html", location_result=location_result)

if __name__=='__main__':
    app.run(debug=True)