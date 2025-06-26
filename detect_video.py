import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime

# 📦 Eğitilmiş YOLOv8 modelini yükle
model = YOLO("runs/arac_plaka_model5/weights/best.pt")

# 🔠 OCR motoru başlat
ocr = easyocr.Reader(['en'])

# 🎥 Video dosyasını aç
video_path = "C:\\Users\\dilek\\Downloads\\2103099-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# 🎯 FPS öğren ve bekleme süresini hesapla (0.5x hız)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int((1000 / fps) * 2)  # 2x bekleme = yarı hız
print(f"FPS: {fps}, 0.5x için gecikme süresi (ms): {delay}")

# 🪟 Pencere ayarları
cv2.namedWindow("Video Test - Araç & Plaka Tanıma", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Test - Araç & Plaka Tanıma", 960, 540)

# Sayaçlar
total_plate_detected = 0
total_ocr_success = 0

# 📄 OCR sonuçlarını kaydedeceğimiz dosya
with open("sonuclar.txt", "a", encoding="utf-8") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes
            vehicle_type = "Bilinmiyor"

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]

                # Kutu ve etiket çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Araç türünü tanı
                if label != "plaka":
                    vehicle_type = label

                # OCR uygula
                if label == "plaka":
                    total_plate_detected += 1
                    roi = frame[y1:y2, x1:x2]
                    ocr_result = ocr.readtext(roi)
                    if ocr_result:
                        total_ocr_success += 1
                        plaka_text = ocr_result[0][1]

                        # Plakayı görüntüye yaz
                        cv2.putText(frame, plaka_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Dosyaya yaz
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] Araç Türü: {vehicle_type} | Plaka: {plaka_text}\n")

        # Görüntüyü %50 küçült (isteğe bağlı)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # Kareyi göster
        cv2.imshow("Video Test - Araç & Plaka Tanıma", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 📊 OCR Performans Özeti
print("\n📊 OCR Performansı")
print(f"Toplam Plaka Tespiti: {total_plate_detected}")
print(f"OCR ile Başarıyla Okunan Plaka: {total_ocr_success}")

if total_plate_detected > 0:
    accuracy = total_ocr_success / total_plate_detected * 100
    print(f"✅ OCR Başarı Oranı: %{accuracy:.2f}")
else:
    print("⚠️ Hiç plaka tespiti yapılamadı.")
