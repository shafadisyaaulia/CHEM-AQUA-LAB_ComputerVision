<div align="center">
  <h1>CHEM-AQUA LAB</h1>
  <h3>Gesture-Controlled Virtual Chemistry Lab</h3>
  <p><em>Proyek UAS Visi Komputer - Simulasi Pencampuran Bahan Kimia Real-time</em></p>
</div>

---

## Deskripsi

**CHEM-AQUA LAB** adalah aplikasi laboratorium kimia virtual berbasis computer vision yang memungkinkan pengguna melakukan simulasi pencampuran bahan kimia secara real-time menggunakan gesture tangan. Proyek ini memanfaatkan MediaPipe Hands untuk deteksi gesture, serta OpenCV untuk visualisasi dan interaksi.

## Fitur Utama

- Deteksi gesture tangan menggunakan MediaPipe
- Simulasi pencampuran bahan kimia dengan perhitungan pH otomatis
- 8+ bahan kimia (asam, basa, garam, indikator)
- Visualisasi hasil campuran, nama senyawa, dan efek partikel
- Edge detection pada cairan
- Kontrol interaktif: tambahkan bahan, reset wadah, dan lihat hasil reaksi
- Panel status campuran: nama reaksi, deskripsi, pH, dan komponen
- Virtual mouse (opsional, file `mouse.py`)

## Instalasi

1. **Clone repository**

	```bash
	git clone https://github.com/shafadisyaaulia/CHEM-AQUA-LAB_ComputerVision.git
	cd CHEM-AQUA-LAB_ComputerVision
	```

2. **Buat virtual environment (opsional tapi disarankan)**

	```bash
	python -m venv .venv
	.venv\Scripts\activate  # Windows
	# source .venv/bin/activate  # Linux/Mac
	```

3. **Install dependencies**

	```bash
	pip install -r requirements.txt
	```

## Cara Menjalankan

### CHEM-AQUA LAB (Simulasi Laboratorium Kimia)

```bash
python ChemAqua-Lab.py
```

- Gerakkan jari telunjuk untuk mengontrol kursor
- Cubit (telunjuk + jempol) untuk klik tombol
- Klik bahan kimia untuk menambah ke wadah
- Klik RESET untuk mengosongkan wadah
- Tekan `q` untuk keluar

### Virtual Mouse (opsional)

```bash
python mouse.py
```

- Kontrol kursor dan klik menggunakan gesture tangan
- Lihat instruksi di jendela aplikasi

## Daftar Dependensi

- opencv-contrib-python==4.10.0.84
- opencv-python==4.10.0.84
- mediapipe==0.10.20
- PyAutoGUI==0.9.54 (untuk virtual mouse)

Semua dependensi dapat diinstal otomatis via `requirements.txt`.

## Struktur Folder

```
├── ChemAqua-Lab.py      # Main aplikasi laboratorium virtual
├── mouse.py             # Virtual mouse berbasis gesture (opsional)
├── requirements.txt     # Daftar dependensi
├── README.md            # Dokumentasi proyek
```

## Kontributor

- Shafa Disya Aulia (shafadisyaaulia)


## Lisensi

Proyek ini dibuat untuk keperluan akademik. Silakan gunakan, modifikasi, dan distribusikan sesuai kebutuhan pembelajaran.

---

<div align="center">
  <sub>UAS Visi Komputer 2025 &copy; CHEM-AQUA LAB</sub>
</div>
