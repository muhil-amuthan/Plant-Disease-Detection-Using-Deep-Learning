from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("plant_disease_model.h5")

# ⚠️ MUST MATCH TRAINING FOLDER ORDER (ALPHABETICAL)
class_names = [
    "Pepper__Bacterial_spot",
    "Pepper__healthy",
    "Potato__Early_blight",
    "Potato__healthy",
    "Potato__Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_YellowLeaf_Curl_Virus"
]

# ================= CROPS =================
CROPS = [
    "Tomato", "Potato", "Pepper"
]

# ================= LANGUAGE =================
LABELS = {
    "en": {
        "title": "Plant Disease Detection",
        "confidence": "Confidence",
        "severity": "Severity",
        "treatment": "Treatment",
        "prevention": "Prevention",
        "back": "New Diagnosis"
    },
    "ta": {
        "title": "தாவர நோய் கண்டறிதல்",
        "confidence": "நம்பகத்தன்மை",
        "severity": "தீவிரம்",
        "treatment": "சிகிச்சை",
        "prevention": "தடுப்பு",
        "back": "மீண்டும்"
    }
}

# ================= HISTORY =================
history = []

# ================= HELPERS =================
def severity(conf):
    if conf >= 85:
        return "High"
    elif conf >= 70:
        return "Medium"
    return "Low"

def treatment_info(disease):
    return {
        "treatment": "Use recommended fungicide or pesticide",
        "prevention": "Remove infected leaves, crop rotation, proper irrigation"
    }

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html", crops=CROPS)

@app.route("/diagnose", methods=["POST"])
def diagnose():
    lang = request.form.get("lang", "en")
    mode = request.form.get("mode")
    crop = request.form.get("crop", "")
    symptoms = request.form.get("symptoms", "")
    file = request.files.get("image")

    disease = "Unknown"
    confidence = 0
    top_predictions = []

    # ---------- IMAGE MODE ----------
    if mode == "image" and file and file.filename != "":
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = image.load_img(path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]

        top_idx = preds.argsort()[-3:][::-1]
        for i in top_idx:
            top_predictions.append({
                "name": class_names[i],
                "confidence": round(float(preds[i] * 100), 2)
            })

        disease = top_predictions[0]["name"]
        confidence = top_predictions[0]["confidence"]

    # ---------- TEXT MODE ----------
    else:
        disease = f"{crop} - Symptom Based Diagnosis"
        confidence = 75
        top_predictions = []

    sev = severity(confidence)
    info = treatment_info(disease)

    history.insert(0, {
        "time": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "crop": crop,
        "disease": disease,
        "confidence": f"{confidence}%"
    })
    history[:] = history[:5]

    return render_template(
        "result.html",
        L=LABELS[lang],
        disease=disease,
        confidence=f"{confidence}%",
        severity=sev,
        treatment=info["treatment"],
        prevention=info["prevention"],
        predictions=top_predictions
    )

@app.route("/history")
def view_history():
    return render_template("history.html", history=history)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
