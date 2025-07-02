# AgriVenture: Smart Agriculture Platform

## Overview
AgriVenture is a comprehensive smart agriculture platform that combines an e-commerce system for agricultural products, prescription uploads, crop and weather prediction, plant disease detection, and IoT-based smart irrigation monitoring. The project leverages Django for the backend, machine learning for crop and disease prediction, and ESP8266-based IoT hardware for real-time field monitoring and automation.

---

## Features

### 1. E-Commerce Platform
- **Product Catalog:** Browse and search for seeds, fertilizers, pesticides, tools, and more.
- **User Authentication:** Signup, login, and secure session management.
- **Cart & Orders:** Add products to cart, checkout, and view order history.
- **Admin Portal:** Manage products, categories, and orders via Django admin.

### 2. Prescription Upload
- **Upload Prescription:** Farmers can upload images of prescriptions for expert review or product recommendations.

### 3. Crop & Weather Prediction
- **Crop Recommendation:** Predicts the best crop to grow based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall using a trained ML model.
- **Weather Forecast:** Integrates with OpenWeatherMap API to provide current and forecasted weather data, including temperature and humidity predictions using ARIMA models.

### 4. Plant Disease Detection
- **Image Upload:** Upload plant leaf images to detect diseases using a deep learning model.
- **Automated Diagnosis:** Returns disease type and suggestions for treatment.

### 5. IoT Smart Irrigation System
- **ESP8266 Integration:** Monitors soil moisture, temperature, humidity, and motion in the field.
- **Blynk Cloud:** Real-time data visualization and control via Blynk mobile app.
- **Automated Water Pump:** Control irrigation based on soil moisture or remotely via app/button.
- **LCD Display:** On-device status updates for field workers.

---

## Project Structure

- `store/` - Django app for e-commerce (products, categories, customers, orders)
- `prescription/` - Handles prescription uploads and ML-powered crop/weather/disease prediction
- `media/` - Stores uploaded images (prescriptions, products, plant leaves)
- `static/` - Static files (CSS, images)
- `jupiternote_book & iot/` - Jupyter notebooks for ML model training and IoT firmware (ESP8266 code)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Eshop-main
```

### 2. Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Django Setup
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

### 4. IoT Firmware
- Flash `BlynkIOT_SmartPlant_Monitoring_Manual.ino` to your ESP8266 device.
- Update Wi-Fi and Blynk credentials as needed.

---

## Usage
- Access the web app at `http://127.0.0.1:8000/`
- Register/login to shop, upload prescriptions, or use prediction tools.
- Use the Blynk app to monitor and control the smart irrigation system.

---

## Requirements
- Python 3.7+
- Django
- scikit-learn, pandas, numpy, ARIMA, etc. (see `requirements.txt`)
- ESP8266 (for IoT)
- Blynk mobile app

---

## Credits
- ML models and IoT code by project contributors.
- Weather data via OpenWeatherMap API.
- Blynk for IoT cloud integration.

---

## License
This project is for educational purposes. 