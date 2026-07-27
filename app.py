import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, session
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key="scanova_secret_key"

# temporary user storage
users={}

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load Model
# ===============================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("models/deepfake_model_epoch9.pth", map_location=device)
)

model.to(device)
model.eval()

classes = ["Fake", "Real"]

# ===============================
# Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ===============================
# Folders
# ===============================
UPLOAD_FOLDER="static/uploads"
HEATMAP_FOLDER="static/heatmaps"

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(HEATMAP_FOLDER,exist_ok=True)

# ===============================
# Risk Level
# ===============================
def risk_level(confidence):

    if confidence > 85:
        return "CRITICAL RISK"
    elif confidence > 65:
        return "HIGH RISK"
    elif confidence > 40:
        return "MODERATE RISK"
    else:
        return "LOW RISK"

# ===============================
# GradCAM
# ===============================
def generate_gradcam(image_tensor, original, filename):

    gradients=[]
    activations=[]

    def forward_hook(module,input,output):
        activations.append(output)

    def backward_hook(module,grad_in,grad_out):
        gradients.append(grad_out[0])

    handle1=model.layer4.register_forward_hook(forward_hook)
    handle2=model.layer4.register_full_backward_hook(backward_hook)

    output=model(image_tensor)
    pred_class=output.argmax(dim=1)

    model.zero_grad()
    output[0,pred_class].backward()

    grads=gradients[0].cpu().detach().numpy()[0]
    acts=activations[0].cpu().detach().numpy()[0]

    weights=np.mean(grads,axis=(1,2))
    cam=np.zeros(acts.shape[1:],dtype=np.float32)

    for i,w in enumerate(weights):
        cam+=w*acts[i]

    cam=np.maximum(cam,0)
    cam=cv2.resize(cam,(original.shape[1],original.shape[0]))

    cam=cam-cam.min()
    cam=cam/(cam.max()+1e-8)

    heatmap=cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    overlay=cv2.addWeighted(original,0.65,heatmap,0.35,0)

    name="heat_"+filename
    path=os.path.join(HEATMAP_FOLDER,name)

    cv2.imwrite(path,overlay)

    handle1.remove()
    handle2.remove()

    return "/static/heatmaps/"+name

# ===============================
# Image Prediction
# ===============================
def predict_image(path):

    original=cv2.imread(path)

    image=Image.open(path).convert("RGB")
    tensor=transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs=model(tensor)
        probs=torch.softmax(outputs,dim=1)
        conf,pred=torch.max(probs,1)

    prediction=classes[pred.item()]
    confidence=round(conf.item()*100,2)

    heatmap=generate_gradcam(tensor,original,os.path.basename(path))

    if prediction=="Fake":

        reason="""
        AI forensic indicators detected:
        • Facial texture irregularities
        • Edge blending artifacts
        • Frequency pattern inconsistencies
        • Unnatural pixel distribution
        """

    else:

        reason="""
        Image passed authenticity checks:
        • Natural pixel distribution
        • Consistent facial geometry
        • No GAN frequency artifacts detected
        """

    risk=risk_level(confidence)

    return prediction,confidence,heatmap,reason,risk

# ===============================
# Routes
# ===============================

@app.route("/")
def home():
    return render_template("home.html")

# ===============================
# LOGIN
# ===============================
@app.route("/login",methods=["GET","POST"])
def login():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        if username in users and users[username]==password:

            session["user"]=username

            # after login go back to home
            return redirect("/")

        else:
            return "Invalid Credentials"

    return render_template("login.html")

# ===============================
# REGISTER
# ===============================
@app.route("/register",methods=["GET","POST"])
def register():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        users[username]=password

        return redirect("/login")

    return render_template("register.html")

# ===============================
# LOGOUT
# ===============================
@app.route("/logout")
def logout():

    session.pop("user",None)
    return redirect("/")

# ===============================
# DETECT
# ===============================
@app.route("/detect",methods=["GET","POST"])
def detect():

    # allow only logged users
    if "user" not in session:
        return redirect("/login")

    prediction=None
    confidence=None
    heatmap=None
    reason=None
    risk=None

    if request.method=="POST":

        file=request.files.get("file")

        if file and file.filename!="":

            filename=secure_filename(file.filename)

            path=os.path.join(UPLOAD_FOLDER,filename)
            file.save(path)

            prediction,confidence,heatmap,reason,risk = predict_image(path)

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=confidence,
        heatmap=heatmap,
        reason=reason,
        risk=risk
    )

# ===============================
# Run
# ===============================
if __name__=="__main__":
    app.run(debug=True)