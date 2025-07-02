#define BLYNK_TEMPLATE_ID "TMPL3NAzmEADM"
#define BLYNK_TEMPLATE_NAME "Smart Irrigation System"
#define BLYNK_AUTH_TOKEN "hSUfcObQ7N9HFEQt-tZ58GsJvu9wXYFc"
// Include necessary libraries
#include <Wire.h>
#include <LiquidCrystal_PCF8574.h>  // ✅ Correct library for ESP8266 LCD
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

// LCD Initialization
LiquidCrystal_PCF8574 lcd(0x3F);

// Wi-Fi Credentials
const char* ssid = "mummy";
const char* pass = "123456789";

// DHT11 Sensor
#define DHTPIN D4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Blynk Timer
BlynkTimer timer;

// Sensor Pins
#define SOIL_MOISTURE_PIN A0
#define PIR_SENSOR_PIN D5
#define RELAY_PIN_1 D3
#define PUSH_BUTTON_1 D7
#define VPIN_BUTTON_1 V0

// Global Variables
int relay1State = LOW;
int pushButton1State = HIGH;
int PIR_Detected = 0;

// 🔹 Function: Connect to Wi-Fi
void connectWiFi() {
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, pass);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi Connected! IP Address: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWi-Fi Connection Failed! Check credentials.");
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("ESP8266 Booting...");

  // Connect to Wi-Fi & Blynk
  connectWiFi();
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass, "blynk.cloud", 80);

  // Initialize LCD
  lcd.begin(16, 2);
  lcd.setBacklight(255);
  lcd.setCursor(0, 0);
  lcd.print("System Booting...");
  delay(2000);
  lcd.clear();

  // Pin Modes
  pinMode(PIR_SENSOR_PIN, INPUT);
  pinMode(RELAY_PIN_1, OUTPUT);
  pinMode(PUSH_BUTTON_1, INPUT_PULLUP);
  digitalWrite(RELAY_PIN_1, relay1State);

  // Initialize Sensors
  dht.begin();
  timer.setInterval(2000L, soilMoistureSensor);
  timer.setInterval(2000L, DHT11sensor);
  timer.setInterval(500L, checkPhysicalButton);
  timer.setInterval(1000L, PIRsensor);  // ✅ Motion detection every 1 sec

  Serial.println("Setup Complete!");
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  delay(2000);
  lcd.clear();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi Disconnected! Reconnecting...");
    connectWiFi();
  }

  Blynk.run();
  timer.run();

  // 🔹 LCD Display for Status Updates
  lcd.setCursor(0, 1);
  lcd.print("M:");
  lcd.print(PIR_Detected ? "ON " : "OFF");

  lcd.setCursor(10, 1);
  lcd.print("W:");
  lcd.print(relay1State ? "ON " : "OFF");
}

// 🔹 Function: Read Soil Moisture
void soilMoistureSensor() {
  int moisture = analogRead(SOIL_MOISTURE_PIN);
  Serial.print("Soil Moisture: ");
  Serial.println(moisture);

  Blynk.virtualWrite(V1, moisture);
  lcd.setCursor(0, 0);
  lcd.print("Soil: ");
  lcd.print(moisture);
}

// 🔹 Function: Read DHT11 Temperature & Humidity
void DHT11sensor() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  
  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read DHT sensor!");
    return;
  }

  Serial.print("Temp: ");
  Serial.print(t);
  Serial.print("°C, Humidity: ");
  Serial.print(h);
  Serial.println("%");

  Blynk.virtualWrite(V2, t);
  Blynk.virtualWrite(V3, h);

  lcd.setCursor(0, 1);
  lcd.print("Temp: ");
  lcd.print(t);
  lcd.print("C ");
}

void PIRsensor() {
  PIR_Detected = digitalRead(PIR_SENSOR_PIN);
  
  if (PIR_Detected == HIGH) {
    Serial.println("Motion Detected!");
    
    // ✅ Correct way to log event in Blynk
    Blynk.logEvent("pir_motion", "⚠️ Motion detected near the field!");

    Blynk.virtualWrite(V4, 1);  // Update Blynk virtual pin
  } else {
    Blynk.virtualWrite(V4, 0);
  }
}


// 🔹 Function: Check Physical Button Press
void checkPhysicalButton() {
  static int lastButtonState = HIGH;
  static unsigned long lastDebounceTime = 0;
  const int debounceDelay = 50;

  int reading = digitalRead(PUSH_BUTTON_1);
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading == LOW) {
      relay1State = !relay1State;
      digitalWrite(RELAY_PIN_1, relay1State);
      Blynk.virtualWrite(VPIN_BUTTON_1, relay1State);
      Serial.print("Water Pump: ");
      Serial.println(relay1State ? "ON" : "OFF");
    }
  }

  lastButtonState = reading;
}
