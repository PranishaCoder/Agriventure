from django.shortcuts import render
from prescription.models import Prescription
from django.http import HttpResponse
from django.contrib import messages
# Create your views here.

def prescrip(request):
    # return render (request, 'upload.html')

# def prescription(request):
    if request.method == "POST":
        fullname = request.POST.get('fullname')
        contactnum = request.POST.get('contactnum')
        image = request.POST.get('image')

        pres = Prescription(fullname = fullname, contactnum = contactnum, image = image)
        pres.save()
        messages.success(request, 'Thankyou.. Your Prescription Submitted Successfuly...')
    return render(request, 'upload.html')

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def my_view(request):
    return HttpResponse("CSRF Disabled Temporarily")





from django.shortcuts import render
import requests
import warnings
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# OpenWeatherMap API Key (Replace with your valid API key)
API_KEY = "3b421e92208e2f94ae07b3f388d96d60"

def index1(request):
    search_done = False
    error_message = None

    if request.method == "POST":
        try:
            city = request.POST['city'].strip()

            # Fetch current weather data
            response = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
            )

            if response.status_code == 404:
                error_message = "City not found. Please enter a valid city."
                return render(request, "index1.html", {"error": error_message})

            response.raise_for_status()
            current_data = response.json()

            # Extract Weather Details
            city_name = current_data['name']
            current_temp = round(current_data['main']['temp'])
            feels_like = round(current_data['main']['feels_like'])
            temp_min = round(current_data['main']['temp_min'])
            temp_max = round(current_data['main']['temp_max'])
            humidity = round(current_data['main']['humidity'])
            country = current_data['sys']['country']
            description = current_data['weather'][0]['description']

            search_done = True

            return render(
                request, "index1.html",
                {"city": city_name, "current_temp": current_temp, "temp_max": temp_max,
                 "temp_min": temp_min, "description": description, "feels_like": feels_like,
                 "country": country, "status": search_done, "humidity": humidity, "error": error_message}
            )

        except requests.exceptions.RequestException as e:
            error_message = f"Error fetching weather data: {e}"
            return render(request, "index1.html", {"error": error_message})

    return render(request, "index1.html", {"status": search_done, "error": error_message})

def predict_weather(request):

    predict_status = False
    error_message = None

    if request.method == "POST":
        try:
            city = request.POST['city'].strip()

            # Fetch future 5-hour weather forecast
            response = requests.get(
                f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
            )

            if response.status_code == 404:
                error_message = "City not found. Please enter a valid city."
                return render(request, "index1.html", {"error": error_message})

            response.raise_for_status()
            forecast_data = response.json()

            # Extract next 5-hour data
            temperature = []
            humidity = []
            time_labels = []
            for i in range(5):
                forecast_time = forecast_data['list'][i]
                temperature.append(round(forecast_time['main']['temp']))
                humidity.append(round(forecast_time['main']['humidity']))
                time_labels.append(datetime.strptime(forecast_time['dt_txt'], "%Y-%m-%d %H:%M:%S").strftime("%H:%M"))

            # Train ARIMA Model
            warnings.filterwarnings("ignore")

            try:
                model_temp = ARIMA(temperature, order=(5,1,0))
                model_temp_fit = model_temp.fit()
                temp_predictions = [round(val) for val in model_temp_fit.forecast(steps=5).tolist()]
            except Exception as e:
                print("Error training ARIMA for temperature:", e)
                temp_predictions = [None] * 5

            try:
                model_hum = ARIMA(humidity, order=(5,1,0))
                model_hum_fit = model_hum.fit()
                hum_predictions = [round(val) for val in model_hum_fit.forecast(steps=5).tolist()]
            except Exception as e:
                print("Error training ARIMA for humidity:", e)
                hum_predictions = [None] * 5

            predict_status = True

            return render(
                request, "index1.html", {"predict_status": predict_status, "city": city,
                                         "temperature_1": temp_predictions[0], "humidity_1": hum_predictions[0],
                                         "temperature_2": temp_predictions[1], "humidity_2": hum_predictions[1],
                                         "temperature_3": temp_predictions[2], "humidity_3": hum_predictions[2],
                                         "temperature_4": temp_predictions[3], "humidity_4": hum_predictions[3],
                                         "temperature_5": temp_predictions[4], "humidity_5": hum_predictions[4],
                                         "tlabels": time_labels, "tvalues": temp_predictions,
                                         "hlabels": time_labels, "hvalues": hum_predictions})

        except requests.exceptions.RequestException as e:
            error_message = f"Error fetching weather data: {e}"
        except KeyError:
            error_message = "Invalid response from API. Please try again."
        except Exception as e:
            error_message = f"Unexpected error occurred: {e}"

        return render(request, "index1.html", {"error": error_message})

    return render(request, "index1.html", {"error": "Invalid request method"})

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import numpy as np
import pickle
import random
import json
import google.generativeai as genai
from sklearn.utils.validation import check_is_fitted

# Configure Gemini
GENAI_API_KEY = "AIzaSyALEqBoYWuylH_4f-WpgR7VFrutrYPQeOQ"
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Load the trained model and scalers
try:
    ml_model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    check_is_fitted(ml_model)
except Exception as e:
    ml_model, sc, ms = None, None, None
    print(f"Error loading model or scalers: {e}")

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

def index2(request):
    return render(request, "index2.html")

def crop_prediction(request):
    if request.method == "POST":
        try:
            N = float(request.POST.get('Nitrogen', 0))
            P = float(request.POST.get('Phosphorus', 0))
            K = float(request.POST.get('Potassium', 0))
            temp = float(request.POST.get('Temperature', 0))
            humidity = float(request.POST.get('Humidity', 0))
            ph = float(request.POST.get('Ph', 0))
            rainfall = float(request.POST.get('Rainfall', 0))

            prompt = (
                f"Based on the following soil and climate data, what is the most suitable crop to grow?\n\n"
                f"Nitrogen: {N} mg/kg\n"
                f"Phosphorus: {P} mg/kg\n"
                f"Potassium: {K} mg/kg\n"
                f"Temperature: {temp}°C\n"
                f"Humidity: {humidity}%\n"
                f"pH: {ph}\n"
                f"Rainfall: {rainfall} mm\n\n"
                f"Give a clear crop name only."
            )

            response = gemini_model.generate_content(prompt)
            crop = response.text.strip()

            result = f"{crop} is the best crop to be cultivated right there."
            return render(request, "index2.html", {"result": result})

        except Exception as e:
            print(f"Prediction Error: {e}")
            return HttpResponse(f"Error in prediction: {str(e)}", status=500)

    return render(request, "index2.html")

def crop_type(request):
    return render(request, "crop_type.html")

def predict1(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            crop = data.get("crop")
            stage = data.get("stage")
            location = data.get("location")
            moisture = data.get("moisture")
            humidity = data.get("humidity")
            temperature = data.get("temperature")

            moisture_levels = [round(random.uniform(60, 80), 2) for _ in range(5)]
            moisture_labels = ["Now", "+1hr", "+2hr", "+3hr", "+4hr"]

            prompt = f"""
You are an agricultural assistant. Based on:
- 🌾 Crop: {crop}
- 🌱 Growth Stage: {stage}
- 📍 Location: {location}
- 💧 Soil Moisture: {moisture}%
- 🌡️ Temperature: {temperature}°C
- 💦 Humidity: {humidity}%

📝 Please provide:
1. Daily water requirement (in mm/day).
2. Reasoning in 2-3 concise, clear points in **English**.
3. Translation of the reasoning in **Marathi**.

Format:
🔹 Water Requirement: X mm/day  
🔸 Reason (English):  
- Point 1  
- Point 2  
🔸 कारण (Marathi):  
- मुद्दा 1  
- मुद्दा 2
"""

            response = gemini_model.generate_content(prompt)

            return JsonResponse({
                "result": response.text.strip(),
                "moisture_levels": moisture_levels,
                "moisture_time_labels": moisture_labels
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed"}, status=405)


import numpy as np
import cv2
from django.shortcuts import render
from io import BytesIO
from PIL import Image

# Try importing Keras model loader
try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

# Load model
def load_plant_disease_model():
    try:
        model = load_model('plant_disease_model.h5')
        return model
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

model = load_plant_disease_model()

CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barley_blight', 'Corn-Common_rust']

# Supplement information for each disease
SUPPLEMENT_INFO = {
    'Tomato-Bacterial_spot': {
        'name': 'Copper-based fungicides',
        'reason': 'Copper fungicides are effective in controlling bacterial infections and prevent the spread of the bacteria.',
        'image': 'data:image/jpeg;base64,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'
    },
    'Potato-Barley_blight': {
        'name': 'Chlorothalonil fungicide',
        'reason': 'Chlorothalonil helps in controlling the spread of blight and prevents damage to potato plants.',
        'image': 'data:image/jpeg;base64,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'
    },
    'Corn-Common_rust': {
        'name': 'Azoxystrobin fungicide',
        'reason': 'Azoxystrobin is effective in controlling fungal diseases like common rust in corn and enhances plant health.',
        'image': 'https://encrypted-tbn2.gstatic.com/shopping?q=tbn:ANd9GcTqRnogJfz6MFF4dH2ILet1PC3Xhv5iU_qZguRW8we-i0viWVtaePH5Vzud8kS0PlwomuNM9lHo5j3ZFsksGpqT0Nj2VGT7c8wwhWh2wWi1iOieIs9z1V-4'
    }
}

def plant_disease_view(request):
    result = None
    supplement_name = None
    supplement_reason = None
    supplement_image = None

    # Only process image if it's uploaded
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        try:
            # Convert the uploaded image to a format usable by OpenCV
            img = Image.open(image)
            img = img.convert('RGB')
            img = np.array(img)
            img = cv2.resize(img, (256, 256))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            if model:
                predictions = model.predict(img)
                label = CLASS_NAMES[np.argmax(predictions[0])]
                result = f"Prediction: This is a {label.split('-')[0]} leaf with {label.split('-')[1].replace('_', ' ')}."

                # Get supplement information based on the prediction
                supplement_name = SUPPLEMENT_INFO.get(label, {}).get('name')
                supplement_reason = SUPPLEMENT_INFO.get(label, {}).get('reason')
                supplement_image = SUPPLEMENT_INFO.get(label, {}).get('image')

            else:
                result = "Model failed to load."

        except Exception as e:
            result = f"Prediction error: {e}"

    # Only return the result if an image was uploaded
    return render(request, 'plant_disease.html', {
        'result': result,
        'supplement_name': supplement_name,
        'supplement_reason': supplement_reason,
        'supplement_image': supplement_image
    })
