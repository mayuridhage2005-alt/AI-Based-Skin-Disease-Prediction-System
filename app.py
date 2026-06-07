from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("skin_model.h5")

class_names = [
    "Acne","Actinic_Keratosis","Benign_tumors","Bullous","Candidiasis",
    "DrugEruption","Eczema","Infestations_Bites","Lichen","Lupus","Moles",
    "Psoriasis","Rosacea","Seborrh_Keratoses","SkinCancer",
    "Sun_Sunlight_Damage","Tinea","Unknown_Normal",
    "Vascular_Tumors","Vasculitis","Vitiligo","Warts"
]

precautions = {
    "Acne":"Wash face gently twice daily, avoid oily products.",
    "Actinic_Keratosis":"Use sunscreen and avoid sun exposure.",
    "Benign_tumors":"Monitor growth and consult dermatologist.",
    "Bullous":"Avoid friction and seek medical care.",
    "Candidiasis":"Keep area dry and use antifungal treatment.",
    "DrugEruption":"Stop suspected drug and consult doctor.",
    "Eczema":"Moisturize regularly and avoid allergens.",
    "Infestations_Bites":"Maintain hygiene and use medication.",
    "Lichen":"Avoid scratching and follow treatment.",
    "Lupus":"Avoid sunlight and follow medical advice.",
    "Moles":"Monitor for color or size changes.",
    "Psoriasis":"Reduce stress and use medicated creams.",
    "Rosacea":"Avoid spicy food and alcohol.",
    "Seborrh_Keratoses":"Usually harmless; consult if irritated.",
    "SkinCancer":"Immediate medical consultation required.",
    "Sun_Sunlight_Damage":"Apply sunscreen and wear protection.",
    "Tinea":"Keep skin dry and apply antifungal cream.",
    "Unknown_Normal":"Skin appears healthy.",
    "Vascular_Tumors":"Consult dermatologist.",
    "Vasculitis":"Seek medical care immediately.",
    "Vitiligo":"Protect skin from sun exposure.",
    "Warts":"Avoid touching and seek treatment if persistent."
}

reasons = {
    "Acne":"Oil and dead skin clog pores causing inflammation.",
    "Actinic_Keratosis":"Long-term UV exposure damages skin cells.",
    "Benign_tumors":"Non-cancerous abnormal skin growth.",
    "Bullous":"Autoimmune reaction causing blisters.",
    "Candidiasis":"Fungal infection in moist areas.",
    "DrugEruption":"Allergic reaction to medicines.",
    "Eczema":"Immune system and environmental triggers.",
    "Infestations_Bites":"Insects or parasites irritate skin.",
    "Lichen":"Immune system attacks skin tissue.",
    "Lupus":"Autoimmune disease affecting skin.",
    "Moles":"Pigment cells cluster together.",
    "Psoriasis":"Rapid skin cell production.",
    "Rosacea":"Inflammation of blood vessels.",
    "Seborrh_Keratoses":"Age-related keratin buildup.",
    "SkinCancer":"DNA damage from UV rays.",
    "Sun_Sunlight_Damage":"Excessive sun exposure.",
    "Tinea":"Fungal infection thrives in warm areas.",
    "Unknown_Normal":"No abnormal condition detected.",
    "Vascular_Tumors":"Abnormal blood vessel growth.",
    "Vasculitis":"Inflammation of blood vessels.",
    "Vitiligo":"Loss of pigment-producing cells.",
    "Warts":"Viral infection (HPV)."
}

# ---------- ROUTES ----------

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        img = request.files["image"]

        if img.filename == "":
            return render_template("home.html")

        # Save image
        img_path = "static/upload.jpg"
        img.save(img_path)

        # Preprocess image
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prediction
        prediction = model.predict(image)
        index = np.argmax(prediction)
        disease = class_names[index]
        confidence = round(float(np.max(prediction)) * 100, 2)

        return render_template(
            "result.html",
            disease=disease.replace("_", " "),
            confidence=confidence,
            precaution=precautions[disease],
            reason=reasons[disease],
            image_path=img_path
        )

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
