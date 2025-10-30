# Optimasi Menu Makanan menggunakan GA dan PSO

Project ini mengimplementasikan optimasi menu makanan seimbang menggunakan dua algoritma: Genetic Algorithm (GA) dan Particle Swarm Optimization (PSO). Tujuannya adalah menemukan kombinasi makanan yang memenuhi kebutuhan nutrisi harian dengan biaya minimal.

## Fitur

- Optimasi menggunakan dua algoritma:
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
- Database 50 makanan Indonesia (10 makanan per kategori)
- Visualisasi perbandingan performa algoritma
- Constraint handling untuk:
  - Target kalori: 1800-2200 kcal
  - Target protein: 50-80g
  - Target karbohidrat: 250-350g
  - Batasan budget: Rp 50.000/hari
  - Porsi minimum per kategori makanan

## Cara Menjalankan

1. Pastikan Python 3.8+ terinstall
2. Clone repository ini
3. Buat virtual environment (opsional tapi direkomendasikan):
```bash
python -m venv .venv
```
4. Aktifkan virtual environment:
- Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```
- Linux/Mac:
```bash
source .venv/bin/activate
```
5. Install dependencies:
```bash
pip install -r requirements.txt
```
6. Jalankan program:
```bash
python optimasi_menu_ga_pso.py
```

Atau gunakan helper script yang tersedia:
- Windows: `.\run.ps1`

## Output

Program akan:
1. Menjalankan optimasi dengan GA dan PSO
2. Menampilkan solusi terbaik dari masing-masing algoritma
3. Membandingkan performa kedua algoritma
4. Menyimpan grafik perbandingan sebagai 'convergence_comparison.png'

## Author

Mochamad Faisal Akbar (L0122094) - Kecerdasan Komputasional