def veritabanina_bagla():
    veri = {
        "sinif": "kamyon",
        "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="guardplate",
            user="postgres",
            password="526352"  # bunu senin gerçek şifrenle değiştir
        )
        cur = conn.cursor()
        cur.execute("INSERT INTO agir_arac (sinif, zaman) VALUES (%s, %s)", (veri["sinif"], veri["zaman"]))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ PostgreSQL'e test verisi başarıyla eklendi.")
    except Exception as e:
        print("❌ Hata:", e)
