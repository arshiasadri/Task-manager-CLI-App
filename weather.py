import tkinter as tk
from http.client import responses
from tkinter import  messagebox
from PIL import Image , ImageTk
import requests
import io
from datetime import datetime

API_KEY = "ENTER"

def get_weather(city_name, api_key):
    url = f"https://www.metaweather.com/api/location/search/?query={city_name}"
    params = {
        "q":city_name,
        "appid": api_key,
        "units": "metric",
        "lang": "fa"
    }
    try:
       response = requests.get(url,params=params)
       data = response.json()
       if response.status_code == 200:
          main = data["main"]
          weather = data["weather"][0]
          icon_code = weather["icon"]
          icon_url = f"http://www.metaweather.com/{icon_code}@2x.png"

          return {
            "city": data["name"],
            "temp": main["temp"],
            "feels_like": main["feels_like"],
            "humidity": main["humidity"],
            "description": weather["description"],
            "icon_url": icon_url,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
          }
       else:
         return None
    except Exception as e:
        print(f"khata:{e}")
        return None
def show_weather():
    city = city_entry.get()
    result = get_weather(city,API_KEY)
    if result:
        info = f"time , date :{result["time"]}\n"
        info += f"city:{result["city"]}\n"
        info += f"description:{result["description"]}\n"
        info += f"temp: {result["temp"]}°C\n"
        info += f"feels_like: {result["feels_like"]}°C\n"
        info += f"humidity: {result["humidity"]}%\n"
        result_label.config(text=info)
        icon_response = requests.get(result["icon_url"])
        icon_data = icon_response.content
        icon_image = Image.open(io.BytesIO(icon_data))
        icon_photo = ImageTk.PhotoImage(icon_image)
        icon_label.config(image=icon_photo)
        icon_label.image = icon_photo
    else:
        result_label.config(text="check your city!!!")
        icon_label.config(image="")
root = tk.Tk()
root.title("weather")
root.geometry("300x400")

tk.Label(root,text="city :", font=("Helvetica",12)).pack(pady=10)
city_entry = tk.Entry(root,font=("Helvetica",12))
city_entry.pack()

tk.Button(root,text="show weather",command=show_weather,font=("Helvetica",12)).pack(pady=10)

icon_label = tk.Label(root)
icon_label.pack()

result_label = tk.Label(root,text="",font=("Helvetica",10),justify="right")
result_label.pack()

root.mainloop()