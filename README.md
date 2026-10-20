# Studi Kasus 1: Make Sense of Data with Analysis and AI

## Deskripsi
Pola PM2.5 harian dan musim kabut asap (data sintetis). Alur: EDA, korelasi, deteksi hari anomali dengan Isolation Forest, regresi linear dari jumlah hotspot dan suhu. Temuan: PM2.5 berkorelasi kuat dengan hotspot (r = 0,76) dan regresi menjelaskan sekitar 70% variasi. Pengembangan: data nyata BMKG, SiPongi KLHK, atau NASA FIRMS.

## Cara menjalankan
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Grafik tersimpan di folder `outputs/`.

## Catatan
Data pada proyek ini sintetis atau bawaan library agar bisa dijalankan tanpa dataset eksternal. Kesimpulan substantif perlu divalidasi dengan data nyata.
