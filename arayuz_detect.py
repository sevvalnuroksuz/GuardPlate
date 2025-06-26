import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr
from datetime import datetime

model = YOLO("weights/best.pt")  # Eğitilmiş modelin yolu
ocr = easyocr.Reader(['en'])

# === GUI Başlat ===
window = tk.Tk()
window.title("Araç ve Plaka Tanıma")
window.geometry("800x600")

video_label = Label(window)
video_label.pack()

plaka_var = tk.StringVar()
Label(window, textvariable=plaka_var, font=("Arial", 14)).pack(pady=10)

cap = None
running = False

def start_camera():
    global cap, running
    cap = cv2.VideoCapture("http://192.168.X.X:8080/video")  # ← Telefon IP adresi!
    running = True
    update_frame()

def stop_camera():
    global running
    running = False
    if cap:
        cap.release()

def update_frame():
    global cap, running
    if not running or not cap:
        return

    ret, frame = cap.read()
    if not ret:
        return

    results = model(frame)

    detected_text = ""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if label == "plaka":
                roi = frame[y1:y2, x1:x2]
                ocr_result = ocr.readtext(roi)
                if ocr_result:
                    plaka = ocr_result[0][1]
                    cv2.putText(frame, plaka, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    detected_text = f"Plaka: {plaka} - Tür: {label}"

                    with open("sonuclar.txt", "a", encoding="utf-8") as f:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp} - {detected_text}\n")

    # Görüntüyü GUI'de göster
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    plaka_var.set(detected_text)

    window.after(10, update_frame)

# === Butonlar ===
Button(window, text="Kamerayı Başlat", command=start_camera).pack(pady=5)
Button(window, text="Durdur", command=stop_camera).pack(pady=5)

# === Çalıştır ===
window.mainloop()
