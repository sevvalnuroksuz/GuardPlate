import tkinter as tk
from tkinter import filedialog, scrolledtext, simpledialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr
from datetime import datetime, time
import psycopg2
import re

model = YOLO(r"C:\\Users\\dilek\\Desktop\\arac_plaka_tanima\\best200epoc.pt")

ocr = easyocr.Reader(['en'])

results_list = []
cap = None
camera_running = False
image_label = None

def saat_araliginda_mi():
    now = datetime.now().time()
    araliklar = [(time(7, 0), time(10, 0)), (time(12, 0), time(14, 0)), (time(16, 0), time(21, 0))]
    return any(start <= now <= end for start, end in araliklar)

def veritabanina_kamyon_ekle(kayit_dict):
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="guardplate",
            user="postgres",
            password="526352"
        )
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO agir_arac (sinif, zaman, plaka) VALUES (%s, %s, %s)",
            (kayit_dict["sinif"], kayit_dict["zaman"], kayit_dict["plaka"])
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ PostgreSQL'e veri eklendi.")
    except Exception as e:
        print("❌ PostgreSQL Hatası:", e)

def process_frame(frame, gui_output=None):
    results = model(frame)
    output_frame = frame.copy()
    for result in results:
        vehicle_type = "Bilinmiyor"
        plaka = ""
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label != "plaka":
                vehicle_type = label

        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, f"{label} {float(box.conf[0]):.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if label == "plaka":
                roi = frame[y1:y2, x1:x2]
                ocr_result = ocr.readtext(roi)
                if ocr_result:
                    plaka = ocr_result[0][1]
                    cv2.putText(output_frame, plaka, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sonuc = f"[{timestamp}] Ara\u00e7: {vehicle_type} | Plaka: [{plaka}]"
                    results_list.append(sonuc)
                    if gui_output:
                        gui_output.insert(tk.END, sonuc + "\n")
    return output_frame

def veritabanina_bagla():
    eklendi = False
    for satir in results_list:
        try:
            if "araç: kamyon" in satir.lower():
                match = re.search(r"\[(.*?)\] Araç: kamyon \| Plaka: (.+)", satir)
                if match:
                    zaman = match.group(1)
                    plaka = match.group(2).strip("[]")  # Çift köşeli parantezleri temizle
                    if saat_araliginda_mi():
                        veri = {
                            "sinif": "kamyon",
                            "zaman": zaman,
                            "plaka": plaka
                        }
                        veritabanina_kamyon_ekle(veri)
                        eklendi = True
                    else:
                        print(f"⏱️ Saat aralık dışında: {zaman}")
                else:
                    print(f"⚠️ Eşleşme sağlanamadı: {satir}")
        except Exception as e:
            print("⚠️ Veri işleme hatası:", e)
    if not eklendi:
        print("🔍 Veritabanına eklenecek uygun kamyon verisi bulunamadı.")


def start_camera(source):
    global cap, camera_running
    cap = cv2.VideoCapture(source)
    camera_running = True
    update_camera_frame()

def update_camera_frame():
    global cap, camera_running
    if cap and cap.isOpened() and camera_running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            processed = process_frame(frame, output_box)
            rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)
        image_label.after(30, update_camera_frame)

def stop_camera():
    global cap, camera_running
    camera_running = False
    if cap:
        cap.release()
    image_label.configure(image='')

def test_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        frame = cv2.imread(path)
        processed = process_frame(frame, output_box)
        rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)

def test_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if path:
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            processed = process_frame(frame, output_box)
            cv2.imshow("Video", processed)
            if cv2.waitKey(30) & 0xFF == ord('q') or cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()

def clear_results():
    results_list.clear()
    output_box.delete(1.0, tk.END)
    stop_camera()

def select_camera():
    def set_source(source_type):
        top.destroy()
        if source_type == 'ip':
            ip_address = simpledialog.askstring("IP Kamera", "IP Kamera URL'si girin:")
            if ip_address:
                start_camera(ip_address)
        else:
            start_camera(0)

    top = tk.Toplevel()
    top.title("Kamera Seçimi")
    tk.Label(top, text="Kamera Türünü Seçin:", font=("Helvetica", 12)).pack(pady=10)
    tk.Button(top, text="Bilgisayar Kamerası", bg="#2196f3", fg="white", width=30, command=lambda: set_source('pc')).pack(pady=5)
    tk.Button(top, text="IP Kamerası", bg="#e8147a", fg="white", width=30, command=lambda: set_source('ip')).pack(pady=5)

def build_gui():
    global output_box, image_label

    root = tk.Tk()
    root.title("Araç & Plaka Tanıma Sistemi")
    root.geometry("1200x700")
    root.configure(bg="#f0f0f0")

    tk.Label(root, text="GuardPlate", font=("Helvetica", 24, "bold"), fg="#333", bg="#f0f0f0").pack(pady=(10, 0))
    tk.Label(root, text="Akıllı Araç Tanıma ve Plaka Okuma Sistemi", font=("Helvetica", 12), fg="#555", bg="#f0f0f0").pack(pady=(0, 10))

    btn_frame = tk.Frame(root, bg="#f0f0f0")
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="📷 Fotoğraflı Test", command=test_image, font=("Helvetica", 12), bg="#ff8b06", fg="white", width=25, height=2).grid(row=0, column=0, padx=5, pady=5)
    tk.Button(btn_frame, text="🎥 Video Test", command=test_video, font=("Helvetica", 12), bg="#4caf50", fg="white", width=25, height=2).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(btn_frame, text="📡 Kamera Test", command=select_camera, font=("Helvetica", 12), bg="#238bec", fg="white", width=25, height=2).grid(row=0, column=2, padx=5, pady=5)
    tk.Button(btn_frame, text="📂 Veritabanına Bağla", command=veritabanina_bagla, font=("Helvetica", 12), bg="#6a1b9a", fg="white", width=25, height=2).grid(row=0, column=3, padx=5, pady=5)
    tk.Button(btn_frame, text="🧹 Temizle", command=clear_results, font=("Helvetica", 12), bg="#f44336", fg="white", width=25, height=2).grid(row=0, column=4, padx=5, pady=5)

    content_frame = tk.Frame(root, bg="#f0f0f0")
    content_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    image_label = tk.Label(content_frame, bg="#ddd", width=600, height=400)
    image_label.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    output_box = scrolledtext.ScrolledText(content_frame, width=50, height=20, font=("Courier", 10))
    output_box.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

build_gui()
