# 🚗 GuardPlate: Akıllı Araç Tanıma ve Plaka Okuma Sistemi

**GuardPlate**, trafik kurallarını ihlal eden ağır taşıtların yasak saatlerde geçişini otomatik olarak tespit etmeyi amaçlayan, yapay zeka destekli bir araç sınıflandırma ve plaka tanıma sistemidir.  
Sistem, **YOLOv8** ile araç tespiti, **EasyOCR** ile plaka okuma işlemleri yapar ve bu verileri **PostgreSQL** veritabanına kaydeder.  
Kullanıcı dostu **Tkinter** arayüzü sayesinde sistem; **kameradan**, **videodan** veya **fotoğraftan** gerçek zamanlı test imkanı sunar.

---

## 🚀 Özellikler

- 📌 **Araç türü sınıflandırma:** Kamyon, otobüs, motosiklet, otomobil ve plaka tespiti (YOLOv8 ile)
- 🔍 **Plaka tanıma:** EasyOCR ile optik karakter tanıma
- 🖥️ **Kullanıcı arayüzü:** Tkinter ile GUI tasarımı
- ⏰ **Zaman kontrolü:** Yasak saatlerde geçiş kontrolü
- 💾 **Veri kaydı:** PostgreSQL ile ihlal kayıtlarının tutulması
- 📷 **Çoklu medya desteği:** Kameradan, videodan ve fotoğraftan test edebilme
- 📡 **Gerçek zamanlı takip:** Araçların canlı takibi ve eş zamanlı analiz

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji     | Açıklama                                       |
|---------------|------------------------------------------------|
| **Python**    | Tüm yazılım geliştirme sürecinde kullanılan dil |
| **YOLOv8**    | Nesne tespiti ve araç türü sınıflandırması için |
| **EasyOCR**   | Plaka karakterlerinin okunması için OCR aracı  |
| **OpenCV**    | Görüntü işleme ve video akışı için             |
| **Tkinter**   | Grafik kullanıcı arayüzü geliştirme             |
| **PostgreSQL**| İhlal verilerinin kayıt altına alınması         |

---

## 📸 Ekran Görüntüleri (Opsiyonel)

![Ekran görüntüsü 2025-06-24 172014](https://github.com/user-attachments/assets/2f375f93-d472-4a8f-8d27-6bae0034981e)



---

## 📂 Proje Çıktıları

- Yasak saatlerde geçen ağır taşıtlar sistem tarafından tespit edilir
- Plaka bilgisi okunarak ihlal kaydı alınır
- Kullanıcı arayüzü üzerinden gerçek zamanlı test yapılabilir
- Tüm veriler veritabanına kayıt edilir ve ileride analiz için kullanılabilir

---

## 🔧 Kurulum ve Kullanım

Kurulum ve kullanım talimatları için `docs/INSTALL.md` veya aşağıdaki adımları takip edebilirsiniz (bu bölüm sizin tercihinize göre eklenebilir):

```bash
# Gerekli kütüphanelerin yüklenmesi
pip install -r requirements.txt

# Uygulamanın çalıştırılması
python app.py

#Test videoları: https://www.youtube.com/playlist?list=PLJrnF7NuefEjmw9RcbovnnBMoImNr2Vcu
