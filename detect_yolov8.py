import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime

# 📦 Eğitilmiş YOLOv8 modelini yükle
model = YOLO("runs\\arac_plaka_model5\\weights\\best.pt")

# 🔠 OCR motoru başlat
ocr = easyocr.Reader(['en'])

# 📷 IP kamera adresi
kamera_url = "http://10.15.176.40:8080/video"
cap = cv2.VideoCapture(kamera_url)

# 🔁 Daha önce yazılan sonucu tutmak için
onceki_kayit = ""

# 🔽 Yardımcı fonksiyon: plaka kutusu araç kutusunun içinde mi?
def plaka_arac_ici_mi(plaka_box, arac_box):
    px1, py1, px2, py2 = plaka_box
    ax1, ay1, ax2, ay2 = arac_box
    return ax1 <= px1 <= ax2 and ay1 <= py1 <= ay2 and ax1 <= px2 <= ax2 and ay1 <= py2 <= ay2

# 📄 OCR sonuçlarını kaydedeceğimiz dosya
with open("sonuclar.txt", "a", encoding="utf-8") as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera bağlantısı başarısız!")
            break

        results = model(frame)

        plaka_list = []
        arac_list = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]

                # Kutu çizimi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]

                if label.lower() == "plaka":
                    ocr_result = ocr.readtext(roi)
                    if ocr_result:
                        plaka_text = ocr_result[0][1]
                        plaka_list.append(((x1, y1, x2, y2), plaka_text))
                        cv2.putText(frame, plaka_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    arac_list.append(((x1, y1, x2, y2), label))

        # Plaka & Araç eşleştir
        for plaka_box, plaka in plaka_list:
            eslesen_tur = "bilinmiyor"
            for arac_box, tur in arac_list:
                if plaka_arac_ici_mi(plaka_box, arac_box):
                    eslesen_tur = tur
                    break  # İlk iç içe olan kutuya eşleştir

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            yeni_kayit = f"{timestamp} - Plaka: {plaka} - Tür: {eslesen_tur}\n"

            if yeni_kayit != onceki_kayit:
                f.write(yeni_kayit)
                print(yeni_kayit.strip())
                onceki_kayit = yeni_kayit

        # Görüntüyü göster
        
        cv2.imshow("Canlı Araç & Plaka Tanıma", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Temizlik
cap.release()
cv2.destroyAllWindows()