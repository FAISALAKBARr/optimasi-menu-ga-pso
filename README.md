# Optimasi Menu Makanan Seimbang

**Genetic Algorithm & Particle Swarm Optimization untuk Perencanaan Menu 4 Sehat 5 Sempurna**

---

## Struktur Project

```
python/
│
├── README.md                          # Dokumentasi ini
├── requirements.txt                   # Dependencies
├── optimasi_menu_ga_pso.py                           # Program utama
├── data/
│   └── food_database.csv             # Dataset (optional)
├── results/
│   ├── convergence_comparison.png    # Hasil visualisasi
│   └── solution_report.txt           # Laporan detail
├── slides/
│   └── proposal.pptx                 # Slide proposal
└── laporan/
    └── laporan_final.pdf             # Laporan IEEE format
```

---

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

### Expected Output

```
==============================================================
OPTIMASI MENU MAKANAN - GA vs PSO
Mochamad Faisal Akbar (L0122094) - Kecerdasan Komputasional
==============================================================

============================================================
GENETIC ALGORITHM
============================================================
Gen   0 | Best Fitness: 15.23 | Cost: Rp 45230 | Cal: 1950
Gen  10 | Best Fitness: 18.45 | Cost: Rp 38450 | Cal: 2010
...
GA Completed in 8.45 seconds

============================================================
PARTICLE SWARM OPTIMIZATION
============================================================
Iter   0 | Best Fitness: 16.12 | Cost: Rp 42100 | Cal: 1980
Iter  10 | Best Fitness: 19.87 | Cost: Rp 36200 | Cal: 2050
...
PSO Completed in 6.23 seconds

Grafik disimpan: convergence_comparison.png
```

---

## Features

### Implemented

- [x] 50 makanan Indonesia (10 per kategori)
- [x] Genetic Algorithm (GA) dengan tournament selection
- [x] Particle Swarm Optimization (PSO) dengan inertia weight
- [x] Multi-constraint handling (nutrisi, budget, kategori)
- [x] Comparative analysis GA vs PSO
- [x] Visualization (convergence curves)
- [x] Detailed solution report

### Objectives

1. **Minimize** total biaya per hari (< Rp 50,000)
2. **Maximize** keseimbangan nutrisi (target AKG)
3. **Ensure** semua kategori 4 Sehat 5 Sempurna terpenuhi

### Constraints

- Kalori: 1800-2200 kcal/hari
- Protein: ≥ 50g/hari
- Karbohidrat: 250-350g/hari
- Setiap kategori minimal 1 porsi
- Budget maksimal: Rp 50,000/hari

---

## Algorithm Parameters

### Genetic Algorithm (GA)

```python
Population Size: 30
Generations: 50
Crossover Rate: 0.8
Mutation Rate: 0.2
Selection: Tournament (k=3)
Elitism: 10%
```

### Particle Swarm Optimization (PSO)

```python
Swarm Size: 20
Iterations: 50
Inertia Weight: 0.9 → 0.4 (linear decay)
Cognitive Coefficient (c1): 2.0
Social Coefficient (c2): 2.0
```

---

## Expected Results

### Performance Comparison

| Metric          | GA          | PSO         |
| --------------- | ----------- | ----------- |
| **Best Cost**   | ~Rp 38,000  | ~Rp 36,000  |
| **Kalori**      | ~2,000 kcal | ~2,050 kcal |
| **Protein**     | ~58g        | ~62g        |
| **Convergence** | ~Gen 35     | ~Iter 28    |
| **Time**        | ~8-10 sec   | ~6-8 sec    |

_Note: Actual results may vary due to stochastic nature_

---

## Dataset

### Sumber Data

1. **Tabel Komposisi Pangan Indonesia (TKPI) 2017**
2. **Survey harga pasar Surakarta (2024)**

### Struktur Dataset (50 makanan)

**Buah (10):** Pisang, Apel, Jeruk, Mangga, Pepaya, Semangka, Anggur, Melon, Pir, Nanas

**Karbohidrat (10):** Nasi, Roti, Mie, Kentang, Singkong, Jagung, Ubi, Pasta, Oatmeal, Roti Gandum

**Protein (10):** Ayam, Telur, Tempe, Tahu, Ikan Lele, Daging Sapi, Ikan Tongkol, Udang, Kacang Merah, Kacang Hijau

**Sayur (10):** Bayam, Kangkung, Wortel, Brokoli, Kol, Tomat, Timun, Terong, Buncis, Labu Siam

**Minuman (10):** Susu Sapi, Teh Manis, Jus Jeruk, Air Kelapa, Susu Kedelai, Yogurt, Kopi Susu, Jus Alpukat, Air Putih, Jus Tomat

---

## How It Works

### 1. Fitness Function

```
Fitness = 1000 / (Total_Cost + Penalty + 1)

Penalty includes:
- Deviation from calorie target
- Protein deficiency
- Carbohydrate deviation
- Budget violation
- Category minimum violation
```

### 2. Solution Representation

```
Solution = [portion_1, portion_2, ..., portion_50]

Each value = portion in grams (0-500g)
Total dimension = 50
```

### 3. Optimization Process

```
1. Initialize population/swarm
2. For each generation/iteration:
   a. Evaluate fitness
   b. Apply operators (selection, crossover, mutation / velocity update)
   c. Update best solution
3. Return best solution found
```

---

## Usage Examples

### Example 1: Run with Default Parameters

```python
python optimasi_menu_ga_pso.py
```

### Example 2: Modify Parameters

```python
# In optimasi_menu_ga_pso.py, modify:
ga = GeneticAlgorithm(
    pop_size=50,      # Increase population
    generations=100,  # More generations
    pc=0.9,          # Higher crossover
    pm=0.1           # Lower mutation
)
```

### Example 3: Export Results

```python
# Add this after running algorithms:
import json

results = {
    'ga_cost': float(ga_nutrition['cost']),
    'pso_cost': float(pso_nutrition['cost']),
    'ga_solution': ga_solution.tolist(),
    'pso_solution': pso_solution.tolist()
}

with open('results/results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Evaluation Metrics

### 1. Solution Quality

- **Total biaya** (Rp/hari)
- **Nutrition score** (% compliance with targets)
- **Constraint violation** (penalty value)

### 2. Algorithm Performance

- **Best fitness** achieved
- **Convergence speed** (generations/iterations to 95%)
- **Computational time** (seconds)

### 3. Statistical Analysis

- Mean ± Standard Deviation (over 10 runs)
- Wilcoxon rank-sum test (p < 0.05)

---

## For Students

### Timeline (5 Weeks)

**Week 1: Setup & Dataset**

- Install Python & libraries
- Review code structure
- Prepare dataset
- Run initial tests

**Week 2: GA Implementation**

- Understand GA concepts
- Modify parameters
- Run experiments
- Analyze results

**Week 3: PSO Implementation**

- Understand PSO concepts
- Modify parameters
- Run experiments
- Compare with GA

**Week 4: Analysis & Comparison**

- Statistical analysis
- Visualization
- Prepare result tables
- Write findings

**Week 5: Report & Presentation**

- Write final report (IEEE format)
- Prepare presentation slides
- Practice presentation
- Submit deliverables

---

## Deliverables Checklist

### Source Code

- [ ] `optimasi_menu_ga_pso.py` - Working implementation
- [ ] Comments & documentation
- [ ] GitHub repository (public)
- [ ] README.md

### Results

- [ ] `convergence_comparison.png` - Visualization
- [ ] `results.txt` - Detailed solution
- [ ] `comparison_table.csv` - GA vs PSO

### Report (IEEE Format)

- [ ] Abstract (150-200 words, 5+ keywords)
- [ ] Introduction (problem, objectives)
- [ ] Methodology (formulation, algorithms)
- [ ] Results & Discussion
- [ ] Conclusion
- [ ] References (IEEE style)

### Presentation

- [ ] Proposal slides (15 minutes)
- [ ] Final presentation slides
- [ ] Demo video (optional)

---

## Troubleshooting

### Issue 1: Import Error

```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:** `pip install numpy pandas matplotlib`

### Issue 2: Slow Execution

```
GA takes >30 seconds
```

**Solution:** Reduce `pop_size` or `generations`

### Issue 3: Poor Results

```
Cost > Rp 60,000 or Kalori < 1500
```

**Solution:**

- Increase iterations/generations
- Tune penalty weights
- Check constraints

---

##References

[1] J. H. Holland, _Adaptation in Natural and Artificial Systems_. University of Michigan Press, 1975.

[2] J. Kennedy and R. Eberhart, "Particle swarm optimization," _Proc. IEEE Int. Conf. Neural Networks_, 1995.

[3] B. K. Seljak, "Computer-based dietary menu planning," _IEEE Trans. Inf. Technol. Biomed._, vol. 13, no. 4, 2009.

[4] Kementerian Kesehatan RI, "Tabel Komposisi Pangan Indonesia 2017," Jakarta, 2017.

---

##Author

**Mochamad Faisal Akbar (L0122094)**  
Program Studi Informatika
Kecerdasan Komputasional - Final Project

---

##Contact

For questions or issues:

- **Email:** faisalzogg022@gmail.com
- **GitHub Issues:** [Create Issue](https://github.com/FAISALAKBARr/optimasi_menu_gas_pso/issues)

---

##License

This project is for educational purposes.  
Dataset sources: TKPI 2017 & local market survey.

---

**Last Updated:** December 2024  
**Version:** 1.0.0
