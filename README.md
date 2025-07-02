# Agriventure : Smart Irrigation System

## Overview

Our solution is a **Smart Irrigation System** with a user-friendly web interface, enabling remote operation. The system uses a **weather API** to monitor real-time conditions like rainfall and temperature, ensuring that the pump operates only when necessary. By integrating **crop type** and **crop size** data, it calculates the precise water requirements for optimal irrigation. This ensures efficient water usage, reduces wastage, and supports sustainable farming practices while improving crop productivity.

## Key Features

1. **AI-Based Plant Disease Prediction**  
   The system uses AI to predict potential plant diseases, allowing users to take preventive actions early.

2. **E-commerce Integration for Agricultural Products**  
   Farmers can purchase essential agricultural products such as seeds, fertilizers, and pesticides through the integrated e-commerce platform.

3. **Weather Updates**  
   Real-time weather updates are provided, helping farmers plan irrigation and farming activities based on current and forecasted weather conditions.

4. **Automated Soil Moisture Analysis**  
   The system performs automated soil moisture analysis to determine the need for irrigation. It advises users to turn the pump **ON** or **OFF** based on real-time soil conditions, preventing overwatering or underwatering.

5. **Crop Type Analysis**  
   The system analyzes the crop type and calculates the amount of water required based on the current soil moisture and the next 5-hour humidity forecast. This ensures that crops receive the exact amount of water needed.

6. **Security Features**  
   The system includes various security features such as **animal deterrence**, **unauthorized access alarms**, and **CCTV monitoring** to protect the crops and farm assets.

7. **Crop Suggestion Based on Soil Analysis**  
   The system provides crop suggestions based on the soil’s health, optimizing farming by recommending the best crops to plant for the current soil conditions.

## Technology Stack

- **Frontend:** HTML, CSS, JavaScript (React.js / Vue.js)
- **Backend:** Django / Flask
- **AI & ML:** Machine Learning Models for disease prediction and crop suggestion
- **Database:** MySQL / PostgreSQL
- **Weather API:** OpenWeatherMap API or similar
- **E-commerce Integration:** Custom API or third-party services
- **Security Features:** CCTV integration, motion detection, and alarm systems

## Installation

Follow these steps to set up the project locally:

1. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

2. **Activate the virtual environment:**

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. **Install the project dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set up the database:**
   Follow the setup instructions in the backend folder to configure the database.

5. **Run the application:**
   For Django:
   ```
   python manage.py runserver
   ```

   Open your browser and visit `http://localhost:8000` (or the port specified) to view the application.

## Contribution

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request.




# plant-disease-prediction-cnn-deep-leanring-project
This repository is about building an Image classifier CNN with Python on Plant Disease Prediction.

Kaggle Dataset Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset



