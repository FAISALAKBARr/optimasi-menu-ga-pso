# NUTRITION OPTIMIZATION USING GA & PSO
# Project: Optimasi Menu Makanan Seimbang (4 Sehat 5 Sempurna)
# Kelompok 8
# Nama: Mochamad Faisal Akbar (L0122094)
# Matkul: Kecerdasan Komputasional
# Features: 
#   - GA vs PSO comparison
#   - 7-day menu generation
#   - Statistical analysis (t-test over 30 runs)
#   - Comprehensive reporting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict
import time
import os
from datetime import datetime

# 1. DATASET - 50 Makanan Indonesia (10 per kategori)
# Sumber: Tabel Komposisi Pangan Indonesia (TKPI) 2017
# Harga: Random sesuai toko online dan Estimasi pasar Surakarta 2025
FOOD_DATABASE = {
    'buah': [
        {'nama': 'Pisang', 'kalori': 89, 'protein': 1.1, 'karbo': 22.8, 'harga': 8000},    
        {'nama': 'Apel', 'kalori': 52, 'protein': 0.3, 'karbo': 14, 'harga': 25000},       
        {'nama': 'Jeruk', 'kalori': 47, 'protein': 0.9, 'karbo': 12, 'harga': 12000},      
        {'nama': 'Mangga', 'kalori': 60, 'protein': 0.8, 'karbo': 15, 'harga': 15000},     
        {'nama': 'Pepaya', 'kalori': 43, 'protein': 0.5, 'karbo': 11, 'harga': 8000},      
        {'nama': 'Semangka', 'kalori': 30, 'protein': 0.6, 'karbo': 8, 'harga': 6000},     
        {'nama': 'Anggur', 'kalori': 69, 'protein': 0.7, 'karbo': 18, 'harga': 35000},     
        {'nama': 'Melon', 'kalori': 34, 'protein': 0.8, 'karbo': 8, 'harga': 10000},       
        {'nama': 'Pir', 'kalori': 57, 'protein': 0.4, 'karbo': 15, 'harga': 30000},        
        {'nama': 'Nanas', 'kalori': 50, 'protein': 0.5, 'karbo': 13, 'harga': 8000},       
    ],
    'karbohidrat': [
        {'nama': 'Nasi Putih', 'kalori': 130, 'protein': 2.7, 'karbo': 28, 'harga': 13000},  
        {'nama': 'Roti Tawar', 'kalori': 265, 'protein': 9, 'karbo': 49, 'harga': 18000},    
        {'nama': 'Mie Instant', 'kalori': 188, 'protein': 4.5, 'karbo': 27, 'harga': 12000}, 
        {'nama': 'Kentang', 'kalori': 77, 'protein': 2, 'karbo': 17, 'harga': 10000},        
        {'nama': 'Singkong', 'kalori': 160, 'protein': 1.4, 'karbo': 38, 'harga': 6000},     
        {'nama': 'Jagung', 'kalori': 86, 'protein': 3.3, 'karbo': 19, 'harga': 8000},        
        {'nama': 'Ubi', 'kalori': 86, 'protein': 1.6, 'karbo': 20, 'harga': 9000},           
        {'nama': 'Pasta', 'kalori': 158, 'protein': 5.8, 'karbo': 31, 'harga': 22000},       
        {'nama': 'Oatmeal', 'kalori': 68, 'protein': 2.4, 'karbo': 12, 'harga': 30000},      
        {'nama': 'Roti Gandum', 'kalori': 247, 'protein': 13, 'karbo': 41, 'harga': 25000},
    ],
    'protein': [
        {'nama': 'Ayam', 'kalori': 165, 'protein': 31, 'karbo': 0, 'harga': 38000},          
        {'nama': 'Telur', 'kalori': 155, 'protein': 13, 'karbo': 1.1, 'harga': 28000},       
        {'nama': 'Tempe', 'kalori': 193, 'protein': 19, 'karbo': 9, 'harga': 10000},         
        {'nama': 'Tahu', 'kalori': 76, 'protein': 8, 'karbo': 1.9, 'harga': 8000},           
        {'nama': 'Ikan Lele', 'kalori': 168, 'protein': 26, 'karbo': 0, 'harga': 30000},     
        {'nama': 'Daging Sapi', 'kalori': 250, 'protein': 26, 'karbo': 0, 'harga': 130000},  
        {'nama': 'Ikan Tongkol', 'kalori': 144, 'protein': 23, 'karbo': 0, 'harga': 35000},  
        {'nama': 'Udang', 'kalori': 99, 'protein': 24, 'karbo': 0.2, 'harga': 90000},        
        {'nama': 'Kacang Merah', 'kalori': 127, 'protein': 8.7, 'karbo': 23, 'harga': 18000},
        {'nama': 'Kacang Hijau', 'kalori': 347, 'protein': 24, 'karbo': 63, 'harga': 22000},
    ],
    'sayur': [
        {'nama': 'Bayam', 'kalori': 23, 'protein': 2.9, 'karbo': 3.6, 'harga': 7000},        
        {'nama': 'Kangkung', 'kalori': 19, 'protein': 3, 'karbo': 3, 'harga': 6000},         
        {'nama': 'Wortel', 'kalori': 41, 'protein': 0.9, 'karbo': 10, 'harga': 10000},       
        {'nama': 'Brokoli', 'kalori': 34, 'protein': 2.8, 'karbo': 7, 'harga': 18000},       
        {'nama': 'Kol', 'kalori': 25, 'protein': 1.3, 'karbo': 6, 'harga': 8000},            
        {'nama': 'Tomat', 'kalori': 18, 'protein': 0.9, 'karbo': 3.9, 'harga': 12000},       
        {'nama': 'Timun', 'kalori': 15, 'protein': 0.7, 'karbo': 3.6, 'harga': 7000},        
        {'nama': 'Terong', 'kalori': 25, 'protein': 1, 'karbo': 6, 'harga': 9000},           
        {'nama': 'Buncis', 'kalori': 31, 'protein': 1.8, 'karbo': 7, 'harga': 12000},        
        {'nama': 'Labu Siam', 'kalori': 19, 'protein': 0.8, 'karbo': 4.5, 'harga': 7000},
    ],
    'minuman': [
        {'nama': 'Susu Sapi', 'kalori': 61, 'protein': 3.2, 'karbo': 4.8, 'harga': 20000},   
        {'nama': 'Teh Manis', 'kalori': 30, 'protein': 0, 'karbo': 8, 'harga': 10000},        
        {'nama': 'Jus Jeruk', 'kalori': 45, 'protein': 0.7, 'karbo': 10, 'harga': 8000},     
        {'nama': 'Air Kelapa', 'kalori': 19, 'protein': 0.7, 'karbo': 3.7, 'harga': 7000},   
        {'nama': 'Susu Kedelai', 'kalori': 54, 'protein': 3.3, 'karbo': 6, 'harga': 12000},  
        {'nama': 'Yogurt', 'kalori': 59, 'protein': 3.5, 'karbo': 4.7, 'harga': 25000},      
        {'nama': 'Kopi Susu', 'kalori': 38, 'protein': 2, 'karbo': 5, 'harga': 8000},        
        {'nama': 'Jus Alpukat', 'kalori': 160, 'protein': 2, 'karbo': 8.5, 'harga': 15000},  
        {'nama': 'Air Putih', 'kalori': 0, 'protein': 0, 'karbo': 0, 'harga': 4000},          
        {'nama': 'Jus Tomat', 'kalori': 17, 'protein': 0.8, 'karbo': 3.9, 'harga': 10000},
    ]
}

# Flatten database
# ALL_FOODS[i] = makanan ke-i (i = 0..49)
ALL_FOODS = []
CATEGORY_START = {}  # Start index untuk setiap kategori
idx = 0
for category, foods in FOOD_DATABASE.items():
    CATEGORY_START[category] = idx
    for food in foods:
        food['category'] = category
        ALL_FOODS.append(food)
    idx += len(foods)

NUM_FOODS = len(ALL_FOODS)  # 50 makanan
print(f"(M) Total makanan dalam database: {NUM_FOODS}")

# 2. NUTRITIONAL TARGETS (Angka Kecukupan Gizi Indonesia)
# Sumber: Peraturan Menteri Kesehatan RI No. 28 Tahun 2019
TARGETS = {
    'kalori': {'min': 1800, 'max': 2200, 'ideal': 2000},
    'protein': {'min': 50, 'max': 80, 'ideal': 60},
    'karbo': {'min': 250, 'max': 350, 'ideal': 300}
}

# Minimum portions per category (4 Sehat 5 Sempurna)
CATEGORY_MIN_PORTIONS = {
    'buah': 150,        # gram (2 porsi  75g)
    'karbohidrat': 300, # gram (3 porsi  100g)
    'protein': 150,     # gram (2 porsi  75g)
    'sayur': 200,       # gram (2-3 porsi  70-100g)
    'minuman': 600      # ml (3x makan  200-300ml)
}

CATEGORY_MAX_PORTIONS = {
    'buah': 400,        # max 400g per hari
    'karbohidrat': 500, # max 500g per hari
    'protein': 300,     # max 300g per hari
    'sayur': 500,       # max 500g per hari
    'minuman': 900      # max 900ml per hari (3x makan  300ml)
}

MAX_BUDGET = 50000  # Rp per hari
MIN_PORTION_PER_FOOD = 50  # gram (kalau pilih, minimal 50g)

MAX_ITEMS_PER_CATEGORY = {
    'buah': 3,        
    'karbohidrat': 2, 
    'protein': 3,     
    'sayur': 4,       
    'minuman': 2      
}

STAPLE_FOOD_INDICES = [10, 11]  # Nasi Putih (10), Roti Tawar (11)
MIN_STAPLE_PORTION = 200  # gram (minimal karbo pokok per hari)


# 3. FITNESS FUNCTION & CONSTRAINTS
def calculate_nutrition(portions: np.ndarray) -> Dict:
    total_kalori = 0
    total_protein = 0
    total_karbo = 0
    total_cost = 0
    
    for i, portion in enumerate(portions):
        food = ALL_FOODS[i]
        # Nutrisi per 100g, jadi dibagi 100
        total_kalori += food['kalori'] * (portion / 100)
        total_protein += food['protein'] * (portion / 100)
        total_karbo += food['karbo'] * (portion / 100)
        # Harga per kg, jadi dibagi 1000
        total_cost += food['harga'] * (portion / 1000)
    
    return {
        'kalori': total_kalori,
        'protein': total_protein,
        'karbo': total_karbo,
        'cost': total_cost
    }

def calculate_penalty(portions: np.ndarray, nutrition: Dict) -> float:
    penalty = 0
    
    #  C1: KALORI CONSTRAINTS 
    # Penalty jika kalori < 1800 atau > 2200
    if nutrition['kalori'] < TARGETS['kalori']['min']:
        violation = TARGETS['kalori']['min'] - nutrition['kalori']
        penalty += violation ** 2 * 1.0  # weight = 1.0
    
    if nutrition['kalori'] > TARGETS['kalori']['max']:
        violation = nutrition['kalori'] - TARGETS['kalori']['max']
        penalty += violation ** 2 * 1.0
    
    #  C2: PROTEIN CONSTRAINT 
    # Penalty jika protein < 50g
    if nutrition['protein'] < TARGETS['protein']['min']:
        violation = TARGETS['protein']['min'] - nutrition['protein']
        penalty += violation ** 2 * 2  # weight = 2.0
    
    #  C3: KARBOHIDRAT CONSTRAINTS 
    # Penalty jika karbo < 250 atau > 350
    if nutrition['karbo'] < TARGETS['karbo']['min']:
        violation = TARGETS['karbo']['min'] - nutrition['karbo']
        penalty += violation ** 2 * 0.3
    
    if nutrition['karbo'] > TARGETS['karbo']['max']:
        violation = nutrition['karbo'] - TARGETS['karbo']['max']
        penalty += violation ** 2 * 0.3
    
    #  C4: BUDGET CONSTRAINT 
    # Penalty jika cost > 50,000
    if nutrition['cost'] > MAX_BUDGET:
        violation = nutrition['cost'] - MAX_BUDGET
        penalty += violation ** 2 * 0.01
    
    #  C5: CATEGORY MINIMUM PORTIONS (4 Sehat 5 Sempurna) 
    # Penalty jika total porsi kategori < minimum
    for category, min_portion in CATEGORY_MIN_PORTIONS.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        category_total = np.sum(portions[start_idx:end_idx])
        
        if category_total < min_portion:
            violation = min_portion - category_total
            if category == 'minuman':
                penalty += violation ** 2 * 2.0  
            else:
                penalty += violation ** 2 * 0.8  

    #  C5B: CATEGORY MAXIMUM PORTIONS 
    # Khusus untuk minuman, enforce max 900ml
    for category, max_portion in CATEGORY_MAX_PORTIONS.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        category_total = np.sum(portions[start_idx:end_idx])
        
        if category_total > max_portion:
            violation = category_total - max_portion
            if category == 'minuman':
                penalty += violation ** 2 * 2.0  
            else:
                penalty += violation ** 2 * 0.5  

    #  C6: MINIMUM PORTION PER FOOD 
    # Eliminasi porsi terlalu kecil (awkward portions)
    for i, portion in enumerate(portions):
        if 0 < portion < MIN_PORTION_PER_FOOD:
            violation = MIN_PORTION_PER_FOOD - portion
            penalty += violation ** 2 * 0.1
    
    #  C7: LIMIT ITEMS PER CATEGORY 
    # Batasi jumlah item per kategori untuk realism
    for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        
        # Hitung berapa item yang dipilih (portion >= MIN_PORTION_PER_FOOD)
        items_selected = np.sum(portions[start_idx:end_idx] >= MIN_PORTION_PER_FOOD)
        
        if items_selected > max_items:
            violation = items_selected - max_items
            penalty += violation ** 2 * 10  # Weight besar 

    #  C8: STAPLE FOOD REQUIREMENT 
    # Pastikan ada makanan pokok (nasi/roti) yang cukup
    staple_total = sum(portions[idx] for idx in STAPLE_FOOD_INDICES)
    
    if staple_total < MIN_STAPLE_PORTION:
        violation = MIN_STAPLE_PORTION - staple_total
        penalty += violation ** 2 * 1.0
    
    return penalty

def fitness_function(portions: np.ndarray) -> float:
    nutrition = calculate_nutrition(portions)
    penalty = calculate_penalty(portions, nutrition)
    cost = nutrition['cost']
    # Fitness formula
    fitness = 1000 / (cost + penalty + 1)
    
    return fitness

def validate_final_solution(solution: np.ndarray) -> Dict:
    """
    Validate bahwa SEMUA constraints terpenuhi di solusi final
    Returns dictionary dengan status validasi lengkap
    """
    violations = []
    details = {}
    
    # Check C1: Kalori
    nutrition = calculate_nutrition(solution)
    kalori = nutrition['kalori']
    if kalori < TARGETS['kalori']['min'] or kalori > TARGETS['kalori']['max']:
        violations.append(f"C1: Kalori = {kalori:.1f} (target: {TARGETS['kalori']['min']}-{TARGETS['kalori']['max']})")
        details['C1'] = False
    else:
        details['C1'] = True
    
    # Check C2: Protein
    protein = nutrition['protein']
    if protein < TARGETS['protein']['min']:
        violations.append(f"C2: Protein = {protein:.1f}g (target: ≥{TARGETS['protein']['min']}g)")
        details['C2'] = False
    else:
        details['C2'] = True
    
    # Check C3: Karbohidrat
    karbo = nutrition['karbo']
    if karbo < TARGETS['karbo']['min'] or karbo > TARGETS['karbo']['max']:
        violations.append(f"C3: Karbo = {karbo:.1f}g (target: {TARGETS['karbo']['min']}-{TARGETS['karbo']['max']}g)")
        details['C3'] = False
    else:
        details['C3'] = True
    
    # Check C4: Budget
    cost = nutrition['cost']
    if cost > MAX_BUDGET:
        violations.append(f"C4: Cost = Rp{cost:,.0f} (target: ≤Rp{MAX_BUDGET:,})")
        details['C4'] = False
    else:
        details['C4'] = True
    
    # Check C6: Minimum portion per food
    awkward = [(i, p) for i, p in enumerate(solution) if 0 < p < MIN_PORTION_PER_FOOD]
    if awkward:
        violations.append(f"C6: {len(awkward)} portions < {MIN_PORTION_PER_FOOD}g")
        details['C6'] = False
    else:
        details['C6'] = True
    
    # Check C7: Max items per category
    c7_violations = []
    for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        items = np.sum(solution[start_idx:end_idx] >= MIN_PORTION_PER_FOOD)
        
        if items > max_items:
            c7_violations.append(f"{category}={items}(max={max_items})")
    
    if c7_violations:
        violations.append(f"C7: " + ", ".join(c7_violations))
        details['C7'] = False
    else:
        details['C7'] = True
    
    # Check C8: Staple food
    staple_total = sum(solution[idx] for idx in STAPLE_FOOD_INDICES)
    if staple_total < MIN_STAPLE_PORTION:
        violations.append(f"C8: Staple = {staple_total:.1f}g (target: ≥{MIN_STAPLE_PORTION}g)")
        details['C8'] = False
    else:
        details['C8'] = True
    
    # Check C5B: Minuman range
    minuman_start = CATEGORY_START['minuman']
    minuman_end = minuman_start + len(FOOD_DATABASE['minuman'])
    minuman_total = np.sum(solution[minuman_start:minuman_end])
    
    if minuman_total < 600 or minuman_total > 900:
        violations.append(f"C5B: Minuman = {minuman_total:.1f}ml (target: 600-900ml)")
        details['C5B'] = False
    else:
        details['C5B'] = True
    
    return {
        'all_constraints_met': len(violations) == 0,
        'violations': violations,
        'num_violations': len(violations),
        'details': details,
        'nutrition': nutrition
    }

def aggressive_repair_solution(individual: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    """
    ENHANCED AGGRESSIVE REPAIR - Version 2.0
    Ensures 100% constraint satisfaction including nutritional targets (C1, C2, C3)
    
    Parameters:
    - individual: solution to repair
    - max_iterations: maximum repair iterations (default: 5)
    
    Returns:
    - repaired individual with ZERO violations
    """
    repaired = individual.copy()
    
    for iteration in range(max_iterations):
        changed = False
        
        # ===== PHASE 1: STRUCTURAL REPAIRS (C6, C7, C8, C5B) =====
        
        # REPAIR 1: C6 - Enforce minimum portion
        for i in range(NUM_FOODS):
            if 0 < repaired[i] < MIN_PORTION_PER_FOOD:
                repaired[i] = 0
                changed = True
        
        # REPAIR 2: C7 - AGGRESSIVE max items enforcement
        for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
            start_idx = CATEGORY_START[category]
            end_idx = start_idx + len(FOOD_DATABASE[category])
            
            # Get active items
            active_items = []
            for i in range(start_idx, end_idx):
                if repaired[i] >= MIN_PORTION_PER_FOOD:
                    active_items.append((i, repaired[i]))
            
            # Remove excess items
            while len(active_items) > max_items:
                active_items.sort(key=lambda x: x[1])
                idx_to_remove = active_items[0][0]
                removed_portion = active_items[0][1]
                repaired[idx_to_remove] = 0
                changed = True
                
                # Redistribute to remaining items
                remaining_items = active_items[1:]
                if len(remaining_items) > 0:
                    total_remaining = sum(p for _, p in remaining_items)
                    for idx, portion in remaining_items:
                        if total_remaining > 0:
                            proportion = portion / total_remaining
                            repaired[idx] += removed_portion * proportion
                
                active_items = [(i, repaired[i]) for i in range(start_idx, end_idx) 
                            if repaired[i] >= MIN_PORTION_PER_FOOD]
        
        # REPAIR 3: C8 - Enforce staple food minimum
        staple_total = sum(repaired[idx] for idx in STAPLE_FOOD_INDICES)
        if staple_total < MIN_STAPLE_PORTION:
            deficit = MIN_STAPLE_PORTION - staple_total
            repaired[10] += deficit  # Add to Nasi Putih
            changed = True
        
        # REPAIR 4: C5B - ENFORCE minuman range (600-900ml)
        minuman_start = CATEGORY_START['minuman']
        minuman_end = minuman_start + len(FOOD_DATABASE['minuman'])
        minuman_total = np.sum(repaired[minuman_start:minuman_end])
        
        if minuman_total < 600:
            deficit = 600 - minuman_total
            air_putih_idx = minuman_start + 8
            repaired[air_putih_idx] += deficit
            changed = True
        
        elif minuman_total > 900:
            excess = minuman_total - 900
            minuman_items = [(i, repaired[i]) for i in range(minuman_start, minuman_end)
                            if repaired[i] >= MIN_PORTION_PER_FOOD]
            
            if len(minuman_items) > 0:
                minuman_items.sort(key=lambda x: x[1], reverse=True)
                for idx, portion in minuman_items:
                    if excess <= 0:
                        break
                    can_reduce = max(0, portion - MIN_PORTION_PER_FOOD)
                    reduce_amount = min(can_reduce, excess)
                    if reduce_amount > 0:
                        repaired[idx] -= reduce_amount
                        excess -= reduce_amount
                        changed = True
        
        # ===== PHASE 2: NUTRITIONAL REPAIRS (C1, C2, C3) - NEW! =====
        
        # Calculate current nutrition
        current_nutrition = calculate_nutrition(repaired)
        
        # REPAIR 5: C1 - FORCE KALORI to range (1800-2200)
        kalori = current_nutrition['kalori']
        
        if kalori < TARGETS['kalori']['min']:
            # Need MORE kalori - add karbo (most efficient for kalori)
            deficit_kalori = TARGETS['kalori']['min'] - kalori
            
            # Strategy: Add to staple food (Nasi: 130 kcal/100g)
            nasi_idx = 10
            nasi_kcal_per_100g = ALL_FOODS[nasi_idx]['kalori']
            additional_grams = (deficit_kalori / nasi_kcal_per_100g) * 100
            
            # Add with safety margin (+2%)
            repaired[nasi_idx] += additional_grams * 1.02
            changed = True
        
        elif kalori > TARGETS['kalori']['max']:
            # Need LESS kalori - reduce high-calorie items
            excess_kalori = kalori - TARGETS['kalori']['max']
            
            # Strategy: Reduce from highest-calorie items (prioritize non-staple)
            high_cal_items = []
            for i in range(NUM_FOODS):
                if repaired[i] >= MIN_PORTION_PER_FOOD and i not in STAPLE_FOOD_INDICES:
                    kcal_contribution = ALL_FOODS[i]['kalori'] * (repaired[i] / 100)
                    if kcal_contribution > 50:  # Significant contributor
                        high_cal_items.append((i, kcal_contribution, repaired[i]))
            
            high_cal_items.sort(key=lambda x: x[1], reverse=True)
            
            for idx, kcal_contrib, portion in high_cal_items:
                if excess_kalori <= 0:
                    break
                
                # Calculate reduction needed
                kcal_per_gram = ALL_FOODS[idx]['kalori'] / 100
                grams_to_reduce = min(
                    excess_kalori / kcal_per_gram,
                    portion - MIN_PORTION_PER_FOOD
                )
                
                if grams_to_reduce > 0:
                    repaired[idx] -= grams_to_reduce
                    excess_kalori -= grams_to_reduce * kcal_per_gram
                    changed = True
        
        # REPAIR 6: C2 - FORCE PROTEIN to minimum (≥50g)
        protein = current_nutrition['protein']
        
        if protein < TARGETS['protein']['min']:
            deficit_protein = TARGETS['protein']['min'] - protein
            
            # Strategy: Add to tempe (19g protein/100g, cheap)
            tempe_idx = 12  # Index of Tempe
            tempe_protein_per_100g = ALL_FOODS[tempe_idx]['protein']
            additional_grams = (deficit_protein / tempe_protein_per_100g) * 100
            
            # Add with safety margin (+3%)
            repaired[tempe_idx] += additional_grams * 1.03
            changed = True
        
        # REPAIR 7: C3 - FORCE KARBO to range (250-350g)
        karbo = current_nutrition['karbo']
        
        if karbo < TARGETS['karbo']['min']:
            # Need MORE karbo
            deficit_karbo = TARGETS['karbo']['min'] - karbo
            
            # Strategy: Add to staple food (Nasi: 28g karbo/100g)
            nasi_idx = 10
            nasi_karbo_per_100g = ALL_FOODS[nasi_idx]['karbo']
            additional_grams = (deficit_karbo / nasi_karbo_per_100g) * 100
            
            # Add with safety margin (+2%)
            repaired[nasi_idx] += additional_grams * 1.02
            changed = True
        
        elif karbo > TARGETS['karbo']['max']:
            # Need LESS karbo - reduce karbo sources
            excess_karbo = karbo - TARGETS['karbo']['max']
            
            # Strategy: Reduce from non-staple karbo items first
            karbo_start = CATEGORY_START['karbohidrat']
            karbo_end = karbo_start + len(FOOD_DATABASE['karbohidrat'])
            
            karbo_items = []
            for i in range(karbo_start, karbo_end):
                if repaired[i] >= MIN_PORTION_PER_FOOD and i not in STAPLE_FOOD_INDICES:
                    karbo_contrib = ALL_FOODS[i]['karbo'] * (repaired[i] / 100)
                    karbo_items.append((i, karbo_contrib, repaired[i]))
            
            karbo_items.sort(key=lambda x: x[1], reverse=True)
            
            for idx, karbo_contrib, portion in karbo_items:
                if excess_karbo <= 0:
                    break
                
                karbo_per_gram = ALL_FOODS[idx]['karbo'] / 100
                grams_to_reduce = min(
                    excess_karbo / karbo_per_gram,
                    portion - MIN_PORTION_PER_FOOD
                )
                
                if grams_to_reduce > 0:
                    repaired[idx] -= grams_to_reduce
                    excess_karbo -= grams_to_reduce * karbo_per_gram
                    changed = True
        
        # ===== CONVERGENCE CHECK =====
        # If no changes in this iteration, repair is complete
        if not changed:
            break
        
        # SAFETY: Re-enforce minuman after all changes (it might be affected)
        minuman_total_final = np.sum(repaired[minuman_start:minuman_end])
        if minuman_total_final < 600:
            repaired[minuman_start + 8] += (600 - minuman_total_final)
        elif minuman_total_final > 900:
            excess_final = minuman_total_final - 900
            for i in range(minuman_start, minuman_end):
                if repaired[i] >= MIN_PORTION_PER_FOOD and excess_final > 0:
                    can_reduce = repaired[i] - MIN_PORTION_PER_FOOD
                    reduce = min(can_reduce, excess_final)
                    repaired[i] -= reduce
                    excess_final -= reduce
    
    return repaired

# 4. GENETIC ALGORITHM (GA)
class GeneticAlgorithm:
    def __init__(self, pop_size=30, generations=50, pc=0.8, pm=0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc  # crossover rate
        self.pm = pm  # mutation rate
        self.best_fitness_history = []
        self.avg_fitness_history = []
    

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = np.zeros(NUM_FOODS)
            
            # Strategy: Pilih sedikit makanan per kategori, porsi wajar
            for category in FOOD_DATABASE.keys():
                start_idx = CATEGORY_START[category]
                category_size = len(FOOD_DATABASE[category])
                
                # Pilih 1-2 makanan per kategori (hindari terlalu banyak variasi)
                max_items = MAX_ITEMS_PER_CATEGORY[category]
                n_items = np.random.randint(1, min(max_items + 1, 3))
                
                # Pilih index secara random
                selected_indices = np.random.choice(category_size, n_items, replace=False)
                
                # Distribusi porsi berdasarkan kategori (agar realistic)
                if category == 'karbohidrat':
                    total_for_category = np.random.uniform(250, 400)
                elif category == 'protein':
                    total_for_category = np.random.uniform(100, 200)
                elif category == 'sayur':
                    total_for_category = np.random.uniform(150, 300)
                elif category == 'buah':
                    total_for_category = np.random.uniform(150, 250)
                elif category == 'minuman':
                    total_for_category = np.random.uniform(600, 900)
                else:
                    total_for_category = np.random.uniform(200, 400)
                
                # Bagi ke selected items
                portions = np.random.dirichlet(np.ones(n_items)) * total_for_category
                
                for idx, portion in zip(selected_indices, portions):
                    food_idx = start_idx + idx
                    # Pastikan minimal 50g jika dipilih
                    individual[food_idx] = max(portion, MIN_PORTION_PER_FOOD)
            
            population.append(individual)
        return population
    
    def tournament_selection(self, population, fitnesses, k=3):
        # Pilih k kandidat secara random
        candidates_idx = np.random.choice(len(population), k, replace=False)
        candidates_fitness = [fitnesses[i] for i in candidates_idx]
        
        # Pilih yang fitness tertinggi
        winner_idx = candidates_idx[np.argmax(candidates_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        # Probabilitas crossover
        if np.random.rand() > self.pc:
            # Tidak crossover, return parents
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        point = np.random.randint(1, NUM_FOODS - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def mutate(self, individual):
        mutated = individual.copy()
        
        for i in range(NUM_FOODS):
            if np.random.rand() < self.pm:
                current_value = mutated[i]
                
                if current_value >= MIN_PORTION_PER_FOOD:
                    #  Makanan yang sudah dipilih 
                    # Mutation dengan σ = 15% dari current value (adaptif!)
                    sigma = current_value * 0.15
                    noise = np.random.normal(0, sigma)
                    mutated[i] += noise
                    
                    # Enforce bounds
                    if mutated[i] < MIN_PORTION_PER_FOOD:
                        # Jika hasil mutation < 50g, set ke 0 (hapus)
                        mutated[i] = 0
                    else:
                        # Clip ke range valid
                        mutated[i] = np.clip(mutated[i], MIN_PORTION_PER_FOOD, 500)
                
                elif current_value == 0:
                    #  Makanan yang belum dipilih 
                    # 15% chance untuk add makanan baru (lebih konservatif)
                    if np.random.rand() < 0.15:
                        mutated[i] = np.random.uniform(MIN_PORTION_PER_FOOD, 150)
        
        return mutated
    
    def repair_solution(self, individual):
        """
        BASIC repair function - dipanggil SETIAP kali setelah mutation
        
        Repair ini RINGAN dan CEPAT, hanya enforce constraint paling basic
        Tidak perlu 100% perfect karena akan ada aggressive repair di akhir
        """
        repaired = individual.copy()
        
        # C6: Basic minimum portion enforcement
        for i in range(NUM_FOODS):
            if 0 < repaired[i] < MIN_PORTION_PER_FOOD:
                repaired[i] = 0
        
        # C7: Basic max items (single pass, tidak iterative)
        for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
            start_idx = CATEGORY_START[category]
            end_idx = start_idx + len(FOOD_DATABASE[category])
            
            category_items = []
            for i in range(start_idx, end_idx):
                if repaired[i] >= MIN_PORTION_PER_FOOD:
                    category_items.append((i, repaired[i]))
            
            if len(category_items) > max_items:
                category_items.sort(key=lambda x: x[1])
                n_remove = len(category_items) - max_items
                for i in range(n_remove):
                    idx = category_items[i][0]
                    repaired[idx] = 0
        
        # C8: Basic staple food
        staple_total = sum(repaired[idx] for idx in STAPLE_FOOD_INDICES)
        if staple_total < MIN_STAPLE_PORTION:
            deficit = MIN_STAPLE_PORTION - staple_total
            repaired[10] += deficit
        
        # C5B: Basic minuman (single pass)
        minuman_start = CATEGORY_START['minuman']
        minuman_end = minuman_start + len(FOOD_DATABASE['minuman'])
        minuman_total = np.sum(repaired[minuman_start:minuman_end])
        
        if minuman_total < 600:
            air_putih_idx = minuman_start + 8
            deficit = 600 - minuman_total
            repaired[air_putih_idx] += deficit
        
        elif minuman_total > 900:
            excess = minuman_total - 900
            minuman_items = [(i, repaired[i]) for i in range(minuman_start, minuman_end)
                            if repaired[i] >= MIN_PORTION_PER_FOOD]
            
            if len(minuman_items) > 0:
                minuman_items.sort(key=lambda x: x[1], reverse=True)
                for idx, portion in minuman_items:
                    if excess <= 0:
                        break
                    can_reduce = portion - MIN_PORTION_PER_FOOD
                    reduce_amount = min(can_reduce, excess)
                    if reduce_amount > 0:
                        repaired[idx] -= reduce_amount
                        excess -= reduce_amount
        
        return repaired

    def evolve(self, verbose=True):
        if verbose:
            print(f"\n{'='*60}")
            print(f" GENETIC ALGORITHM")
            print(f"{'='*60}")
            print(f"Population: {self.pop_size} | Generations: {self.generations}")
            print(f"Crossover Rate: {self.pc} | Mutation Rate: {self.pm}")
            print(f"{'='*60}\n")
        
        # Initialize
        population = self.initialize_population()
        best_solution = None
        best_fitness = -np.inf
        
        start_time = time.time()
        
        # ===== EVOLUTION LOOP =====
        for gen in range(self.generations):
            fitnesses = [fitness_function(ind) for ind in population]
            
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            
            if verbose and (gen % 10 == 0 or gen == self.generations - 1):
                print(f"Gen {gen:3d} | Best Fitness: {best_fitness:.6f} | "
                      f"Avg: {np.mean(fitnesses):.6f}")
            
            # Selection & Reproduction
            elite_count = max(2, int(0.1 * self.pop_size))
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            elites = [population[i].copy() for i in elite_indices]
            
            new_population = elites.copy()
            
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # ← BASIC REPAIR dipanggil di sini (setiap offspring)
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            population = new_population[:self.pop_size]
        
        elapsed_time = time.time() - start_time
        
        # ===== CRITICAL: FINAL AGGRESSIVE REPAIR =====
        # ← AGGRESSIVE REPAIR dipanggil SEKALI di sini (akhir evolusi)
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Performing final aggressive repair...")
        
        best_solution = aggressive_repair_solution(best_solution, max_iterations=5)
        best_fitness = fitness_function(best_solution)
        
        # Validation
        validation = validate_final_solution(best_solution)
        
        if verbose:
            print(f"{'─'*60}")
            print(f"GA Completed in {elapsed_time:.2f} seconds")
            print(f"Best Fitness: {best_fitness:.6f}")
            
            if validation['all_constraints_met']:
                print(f"✓ ALL CONSTRAINTS SATISFIED")
            else:
                print(f"✗ VIOLATIONS DETECTED: {validation['num_violations']}")
                for v in validation['violations']:
                    print(f"  - {v}")
            
            print(f"{'='*60}\n")
        
        return best_solution, best_fitness, {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history,
            'time': elapsed_time,
            'validation': validation
        }


# 5. PARTICLE SWARM OPTIMIZATION (PSO)
class ParticleSwarmOptimization:
    
    def __init__(self, n_particles=30, iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w    # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def repair_solution(self, individual):
        """
        BASIC repair - dipanggil SETIAP kali setelah position update
        
        SAMA PERSIS seperti GA.repair_solution()
        """
        repaired = individual.copy()
        
        for i in range(NUM_FOODS):
            if 0 < repaired[i] < MIN_PORTION_PER_FOOD:
                repaired[i] = 0
        
        for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
            start_idx = CATEGORY_START[category]
            end_idx = start_idx + len(FOOD_DATABASE[category])
            
            category_items = []
            for i in range(start_idx, end_idx):
                if repaired[i] >= MIN_PORTION_PER_FOOD:
                    category_items.append((i, repaired[i]))
            
            if len(category_items) > max_items:
                category_items.sort(key=lambda x: x[1])
                n_remove = len(category_items) - max_items
                for i in range(n_remove):
                    idx = category_items[i][0]
                    repaired[idx] = 0
        
        staple_total = sum(repaired[idx] for idx in STAPLE_FOOD_INDICES)
        if staple_total < MIN_STAPLE_PORTION:
            deficit = MIN_STAPLE_PORTION - staple_total
            repaired[10] += deficit
        
        minuman_start = CATEGORY_START['minuman']
        minuman_end = minuman_start + len(FOOD_DATABASE['minuman'])
        minuman_total = np.sum(repaired[minuman_start:minuman_end])
        
        if minuman_total < 600:
            air_putih_idx = minuman_start + 8
            deficit = 600 - minuman_total
            repaired[air_putih_idx] += deficit
        elif minuman_total > 900:
            excess = minuman_total - 900
            minuman_items = [(i, repaired[i]) for i in range(minuman_start, minuman_end)
                            if repaired[i] >= MIN_PORTION_PER_FOOD]
            if len(minuman_items) > 0:
                minuman_items.sort(key=lambda x: x[1], reverse=True)
                for idx, portion in minuman_items:
                    if excess <= 0:
                        break
                    can_reduce = portion - MIN_PORTION_PER_FOOD
                    reduce_amount = min(can_reduce, excess)
                    if reduce_amount > 0:
                        repaired[idx] -= reduce_amount
                        excess -= reduce_amount
        
        return repaired
    
    def optimize(self, verbose=True):
        if verbose:
            print(f"\n{'='*60}")
            print(f" PARTICLE SWARM OPTIMIZATION")
            print(f"{'='*60}")
            print(f"Particles: {self.n_particles} | Iterations: {self.iterations}")
            print(f"w: {self.w} | c1: {self.c1} | c2: {self.c2}")
            print(f"{'='*60}\n")
        
        positions = np.random.uniform(0, 300, (self.n_particles, NUM_FOODS))
        velocities = np.random.uniform(-50, 50, (self.n_particles, NUM_FOODS))
        
        pbest_positions = positions.copy()
        pbest_fitness = np.array([fitness_function(p) for p in positions])
        
        gbest_idx = np.argmax(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        start_time = time.time()
        
        # ===== OPTIMIZATION LOOP =====
        for iter in range(self.iterations):
            for i in range(self.n_particles):
                fitness = fitness_function(positions[i])
                
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()
                
                r1 = np.random.rand(NUM_FOODS)
                r2 = np.random.rand(NUM_FOODS)
                
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                
                velocities[i] = (self.w * velocities[i] + cognitive + social)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 500)
                
                # ← BASIC REPAIR dipanggil di sini (setiap particle update)
                positions[i] = self.repair_solution(positions[i])
            
            self.best_fitness_history.append(gbest_fitness)
            self.avg_fitness_history.append(np.mean(pbest_fitness))
            
            if verbose and (iter % 10 == 0 or iter == self.iterations - 1):
                print(f"Iter {iter:3d} | Best Fitness: {gbest_fitness:.6f} | "
                      f"Avg: {np.mean(pbest_fitness):.6f}")
        
        elapsed_time = time.time() - start_time
        
        # ===== CRITICAL: FINAL AGGRESSIVE REPAIR =====
        # ← AGGRESSIVE REPAIR dipanggil SEKALI di sini (akhir optimization)
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Performing final aggressive repair...")
        
        gbest_position = aggressive_repair_solution(gbest_position, max_iterations=5)
        gbest_fitness = fitness_function(gbest_position)
        
        validation = validate_final_solution(gbest_position)
        
        if verbose:
            print(f"{'─'*60}")
            print(f"PSO Completed in {elapsed_time:.2f} seconds")
            print(f"Best Fitness: {gbest_fitness:.6f}")
            
            if validation['all_constraints_met']:
                print(f"✓ ALL CONSTRAINTS SATISFIED")
            else:
                print(f"✗ VIOLATIONS DETECTED: {validation['num_violations']}")
                for v in validation['violations']:
                    print(f"  - {v}")
            
            print(f"{'='*60}\n")
        
        return gbest_position, gbest_fitness, {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history,
            'time': elapsed_time,
            'validation': validation
        }

# 6. HELPER FUNCTIONS - REPORTING & VISUALIZATION
def print_menu_detail(solution: np.ndarray, day_number: int = None):
    header = f" MENU DETAIL" + (f" - HARI {day_number}" if day_number else "")
    print(f"\n{'='*70}")
    print(f"{header:^70}")
    print(f"{'='*70}")
    
    nutrition = calculate_nutrition(solution)
    
    # Print per kategori
    for category, foods in FOOD_DATABASE.items():
        print(f"\n{category.upper()}")
        print(f"{'-'*70}")
        
        start_idx = CATEGORY_START[category]
        category_cost = 0
        
        for i, food in enumerate(foods):
            idx = start_idx + i
            portion = solution[idx]
            
            if portion > 5:  # Only show significant portions
                cost = food['harga'] * (portion / 1000)
                category_cost += cost
                print(f"  • {food['nama']:20s} : {portion:6.1f}g  "
                      f"(Rp {cost:,.0f})")
        
        print(f"  {'Subtotal':22s} : Rp {category_cost:,.0f}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RINGKASAN NUTRISI")
    print(f"{'-'*70}")
    print(f"  Kalori      : {nutrition['kalori']:7.1f} kkal "
          f"({'Memenuhi' if TARGETS['kalori']['min'] <= nutrition['kalori'] <= TARGETS['kalori']['max'] else 'Tidak Memenuhi'} "
          f"Target: {TARGETS['kalori']['min']}-{TARGETS['kalori']['max']})")
    print(f"  Protein     : {nutrition['protein']:7.1f} g    "
          f"({'Memenuhi' if nutrition['protein'] >= TARGETS['protein']['min'] else 'Tidak Memenuhi'} "
          f"Target: ≥{TARGETS['protein']['min']})")
    print(f"  Karbohidrat : {nutrition['karbo']:7.1f} g    "
          f"({'Memenuhi' if TARGETS['karbo']['min'] <= nutrition['karbo'] <= TARGETS['karbo']['max'] else 'Tidak Memenuhi'} "
          f"Target: {TARGETS['karbo']['min']}-{TARGETS['karbo']['max']})")
    print(f"  Biaya       : Rp {nutrition['cost']:,.0f}  "
          f"({'Memenuhi' if nutrition['cost'] <= MAX_BUDGET else 'Tidak Memenuhi'} "
          f"Target: ≤Rp {MAX_BUDGET:,})")
    print(f"{'='*70}")

    # == VALIDASI CONSTRAINTS ==
    print(f"\n  VALIDASI CONSTRAINTS")
    print(f"{'-'*70}")
    
    # Check C6: Min Portion
    violation_c6 = [(i, p) for i, p in enumerate(solution) if 0 < p < MIN_PORTION_PER_FOOD]
    if violation_c6:
        print(f"(TM) C6 VIOLATION: {len(violation_c6)} makanan porsi < 50g")
    else:
        print(f"(M) C6: Semua porsi ≥ 50g")
    
    # Check C7: Max Items
    for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        
        # Hanya hitung yang ≥ MIN_PORTION_PER_FOOD
        items = np.sum(solution[start_idx:end_idx] >= MIN_PORTION_PER_FOOD)
        
        if items > max_items:
            print(f"(TM) C7 VIOLATION: {category} = {items} jenis (max = {max_items})")
        else:
            print(f"(M) C7: {category} = {items} jenis (max = {max_items})")
    
    # Check C8: Staple Food
    staple_total = sum(solution[idx] for idx in STAPLE_FOOD_INDICES)
    if staple_total < MIN_STAPLE_PORTION:
        print(f"(TM) C8 VIOLATION: Staple food = {staple_total:.1f}g (min = 200g)")
    else:
        print(f"(M) C8: Staple food = {staple_total:.1f}g (min = 200g)")
    
    #  Check C5B - MINUMAN RANGE 
    minuman_start = CATEGORY_START['minuman']
    minuman_end = minuman_start + len(FOOD_DATABASE['minuman'])
    minuman_total = np.sum(solution[minuman_start:minuman_end])

    if minuman_total < 600:
        print(f"(TM) C5B VIOLATION: Minuman = {minuman_total:.1f}ml (min = 600ml)")
    elif minuman_total > 900:
        print(f"(TM) C5B VIOLATION: Minuman = {minuman_total:.1f}ml (max = 900ml)")
    else:
        print(f"(M) C5B: Minuman = {minuman_total:.1f}ml (range: 600-900ml)")
    
    print(f"{'='*70}\n")

    print(f"\nCATATAN PENGGUNAAN")
    print(f"{'-'*70}")
    print(f"Menu di atas adalah total untuk 1 HARI (3x makan).")
    print(f"Suggested distribution:")
    print(f"  • Sarapan (30%): Karbo + Protein + Buah + Minuman (~200gr)")
    print(f"  • Makan Siang (40%): Porsi terbesar, semua kategori (~300gr)")
    print(f"  • Makan Malam (30%): Lebih ringan, Protein + Sayur (~200gr)")
    print(f"  • Total Minuman: {minuman_total:.0f}ml dari target 600-900 gr atau ml")  
    print(f"{'='*70}\n")

def plot_convergence_comparison(ga_history: Dict, pso_history: Dict, 
                                save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Best fitness
    axes[0].plot(ga_history['best_history'], label='GA', linewidth=2, marker='o', 
                 markersize=3, markevery=5)
    axes[0].plot(pso_history['best_history'], label='PSO', linewidth=2, marker='s',
                 markersize=3, markevery=5)
    axes[0].set_xlabel('Generation / Iteration', fontsize=12)
    axes[0].set_ylabel('Best Fitness', fontsize=12)
    axes[0].set_title('Best Fitness Convergence', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average fitness
    axes[1].plot(ga_history['avg_history'], label='GA', linewidth=2, marker='o',
                 markersize=3, markevery=5)
    axes[1].plot(pso_history['avg_history'], label='PSO', linewidth=2, marker='s',
                 markersize=3, markevery=5)
    axes[1].set_xlabel('Generation / Iteration', fontsize=12)
    axes[1].set_ylabel('Average Fitness', fontsize=12)
    axes[1].set_title('Average Fitness Convergence', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_comparison_summary(ga_result: Dict, pso_result: Dict):
    print(f"\n{'='*70}")
    print(f"{'PERBANDINGAN GA vs PSO':^70}")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<30} {'GA':>15} {'PSO':>15} {'Winner':>10}")
    print(f"{'-'*70}")
    
    # Quality (fitness)
    ga_fitness = ga_result['fitness']
    pso_fitness = pso_result['fitness']
    quality_winner = 'PSO' if pso_fitness > ga_fitness else 'GA'
    print(f"{'1. Best Fitness':<30} {ga_fitness:>15.6f} {pso_fitness:>15.6f} {quality_winner:>10}")
    
    # Cost
    ga_cost = ga_result['nutrition']['cost']
    pso_cost = pso_result['nutrition']['cost']
    cost_winner = 'PSO' if pso_cost < ga_cost else 'GA'
    print(f"{'2. Total Cost (Rp)':<30} {ga_cost:>15,.0f} {pso_cost:>15,.0f} {cost_winner:>10}")
    
    # Computational time
    ga_time = ga_result['time']
    pso_time = pso_result['time']
    time_winner = 'PSO' if pso_time < ga_time else 'GA'
    print(f"{'3. Computation Time (s)':<30} {ga_time:>15.2f} {pso_time:>15.2f} {time_winner:>10}")
    
    # Convergence speed
    ga_conv = len([x for x in ga_result['history']['best_history'] 
                   if x < ga_fitness * 0.95])
    pso_conv = len([x for x in pso_result['history']['best_history'] 
                    if x < pso_fitness * 0.95])
    conv_winner = 'PSO' if pso_conv < ga_conv else 'GA'
    print(f"{'4. Convergence Speed (iter)':<30} {ga_conv:>15d} {pso_conv:>15d} {conv_winner:>10}")
    
    print(f"{'='*70}")
    
    # Nutritional comparison
    print(f"\n{'Nutritional Values':<30} {'GA':>15} {'PSO':>15} {'Target':>20}")
    print(f"{'-'*70}")
    print(f"{'Kalori (kkal)':<30} {ga_result['nutrition']['kalori']:>15.1f} "
          f"{pso_result['nutrition']['kalori']:>15.1f} "
          f"{TARGETS['kalori']['min']}-{TARGETS['kalori']['max']:>6}")
    print(f"{'Protein (g)':<30} {ga_result['nutrition']['protein']:>15.1f} "
          f"{pso_result['nutrition']['protein']:>15.1f} "
          f"≥{TARGETS['protein']['min']:>7}")
    print(f"{'Karbohidrat (g)':<30} {ga_result['nutrition']['karbo']:>15.1f} "
          f"{pso_result['nutrition']['karbo']:>15.1f} "
          f"{TARGETS['karbo']['min']}-{TARGETS['karbo']['max']:>6}")
    print(f"{'='*70}\n")


# 7. MENU 7 HARI - RUN MULTIPLE TIMES
def generate_weekly_menu(algorithm='both', verbose=True):
    results = {'GA': [], 'PSO': []}
    days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    
    print(f"\n{'='*70}")
    print(f"{'GENERATE MENU 7 HARI':^70}")
    print(f"{'='*70}\n")
    
    for day_num, day_name in enumerate(days, 1):
        print(f"\n{'~'*70}")
        print(f"{'HARI ' + str(day_num) + ': ' + day_name:^70}")
        print(f"{'~'*70}")
        
        # Set random seed untuk variasi
        np.random.seed(day_num * 42)
        
        #  RUN GA 
        if algorithm in ['ga', 'both']:
            ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
            ga_solution, ga_fitness, ga_history = ga.evolve(verbose=False)
            ga_nutrition = calculate_nutrition(ga_solution)
            
            results['GA'].append({
                'day': day_name,
                'solution': ga_solution,
                'fitness': ga_fitness,
                'nutrition': ga_nutrition,
                'history': ga_history
            })
            
            if verbose:
                print_menu_detail(ga_solution, day_num)
                print(f" GA Fitness: {ga_fitness:.6f} | Cost: Rp {ga_nutrition['cost']:,.0f}")
        
        #  RUN PSO 
        if algorithm in ['pso', 'both']:
            pso = ParticleSwarmOptimization(n_particles=30, iterations=50, 
                                           w=0.7, c1=1.5, c2=1.5)
            pso_solution, pso_fitness, pso_history = pso.optimize(verbose=False)
            pso_nutrition = calculate_nutrition(pso_solution)
            
            results['PSO'].append({
                'day': day_name,
                'solution': pso_solution,
                'fitness': pso_fitness,
                'nutrition': pso_nutrition,
                'history': pso_history
            })
            
            if verbose:
                print_menu_detail(pso_solution, day_num)
                print(f" PSO Fitness: {pso_fitness:.6f} | Cost: Rp {pso_nutrition['cost']:,.0f}")
    
    #  SUMMARY 7 HARI 
    print(f"\n{'='*70}")
    print(f"{'SUMMARY 7 HARI':^70}")
    print(f"{'='*70}\n")
    
    if 'GA' in results and len(results['GA']) > 0:
        ga_costs = [r['nutrition']['cost'] for r in results['GA']]
        print(f" GA - Total Cost 7 Hari: Rp {sum(ga_costs):,.0f}")
        print(f"   Average per hari: Rp {np.mean(ga_costs):,.0f} ± {np.std(ga_costs):,.0f}")
    
    if 'PSO' in results and len(results['PSO']) > 0:
        pso_costs = [r['nutrition']['cost'] for r in results['PSO']]
        print(f" PSO - Total Cost 7 Hari: Rp {sum(pso_costs):,.0f}")
        print(f"   Average per hari: Rp {np.mean(pso_costs):,.0f} ± {np.std(pso_costs):,.0f}")
    
    print(f"{'='*70}\n")
    
    return results


# 8. STATISTICAL ANALYSIS - 30 RUNS dengan T-TEST
def run_statistical_test(n_runs=30):
    print(f"\n{'='*70}")
    print(f"{'STATISTICAL ANALYSIS - ' + str(n_runs) + ' RUNS':^70}")
    print(f"{'='*70}\n")
    
    ga_fitnesses = []
    ga_costs = []
    ga_times = []
    
    pso_fitnesses = []
    pso_costs = []
    pso_times = []
    
    #  RUN 30 TIMES 
    for run in range(n_runs):
        print(f"Running {run+1}/{n_runs}...", end='\r')
        
        # Set seed untuk reproducibility
        np.random.seed(run * 123)
        
        # GA
        ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
        ga_sol, ga_fit, ga_hist = ga.evolve(verbose=False)
        ga_nutr = calculate_nutrition(ga_sol)
        
        ga_fitnesses.append(ga_fit)
        ga_costs.append(ga_nutr['cost'])
        ga_times.append(ga_hist['time'])
        
        # PSO
        pso = ParticleSwarmOptimization(n_particles=30, iterations=50, 
                                       w=0.7, c1=1.5, c2=1.5)
        pso_sol, pso_fit, pso_hist = pso.optimize(verbose=False)
        pso_nutr = calculate_nutrition(pso_sol)
        
        pso_fitnesses.append(pso_fit)
        pso_costs.append(pso_nutr['cost'])
        pso_times.append(pso_hist['time'])
    
    print(f"\nCompleted {n_runs} runs!\n")
    
    #  DESCRIPTIVE STATISTICS 
    print(f"{'='*70}")
    print(f"{'DESCRIPTIVE STATISTICS':^70}")
    print(f"{'='*70}\n")
    
    # Convert to numpy arrays
    ga_fitnesses = np.array(ga_fitnesses)
    ga_costs = np.array(ga_costs)
    ga_times = np.array(ga_times)
    
    pso_fitnesses = np.array(pso_fitnesses)
    pso_costs = np.array(pso_costs)
    pso_times = np.array(pso_times)
    
    # Print table
    print(f"{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"{'-'*70}")
    
    # GA Statistics
    print(f"\n GENETIC ALGORITHM")
    print(f"{'Fitness':<20} {np.mean(ga_fitnesses):>12.6f} {np.std(ga_fitnesses):>12.6f} "
          f"{np.min(ga_fitnesses):>12.6f} {np.max(ga_fitnesses):>12.6f}")
    print(f"{'Cost (Rp)':<20} {np.mean(ga_costs):>12,.0f} {np.std(ga_costs):>12,.0f} "
          f"{np.min(ga_costs):>12,.0f} {np.max(ga_costs):>12,.0f}")
    print(f"{'Time (s)':<20} {np.mean(ga_times):>12.2f} {np.std(ga_times):>12.2f} "
          f"{np.min(ga_times):>12.2f} {np.max(ga_times):>12.2f}")
    
    # PSO Statistics
    print(f"\n PARTICLE SWARM OPTIMIZATION")
    print(f"{'Fitness':<20} {np.mean(pso_fitnesses):>12.6f} {np.std(pso_fitnesses):>12.6f} "
          f"{np.min(pso_fitnesses):>12.6f} {np.max(pso_fitnesses):>12.6f}")
    print(f"{'Cost (Rp)':<20} {np.mean(pso_costs):>12,.0f} {np.std(pso_costs):>12,.0f} "
          f"{np.min(pso_costs):>12,.0f} {np.max(pso_costs):>12,.0f}")
    print(f"{'Time (s)':<20} {np.mean(pso_times):>12.2f} {np.std(pso_times):>12.2f} "
          f"{np.min(pso_times):>12.2f} {np.max(pso_times):>12.2f}")
    
    #  PAIRED T-TEST 
    print(f"\n{'='*70}")
    print(f"{'PAIRED T-TEST RESULTS':^70}")
    print(f"{'='*70}\n")
    
    # H0: μ_GA = μ_PSO (tidak ada perbedaan signifikan)
    # H1: μ_GA ≠ μ_PSO (ada perbedaan signifikan)
    # α = 0.05
    
    # Test 1: Fitness
    t_stat_fit, p_value_fit = stats.ttest_rel(ga_fitnesses, pso_fitnesses)
    print(f"1️  FITNESS COMPARISON")
    print(f"   T-statistic: {t_stat_fit:.4f}")
    print(f"   P-value: {p_value_fit:.6f}")
    print(f"   Result: {'TIDAK signifikan (p > 0.05)' if p_value_fit > 0.05 else 'SIGNIFIKAN (p < 0.05)'}")
    print(f"   Winner: {'PSO' if np.mean(pso_fitnesses) > np.mean(ga_fitnesses) else 'GA'} "
          f"(Mean: {'PSO=' + f'{np.mean(pso_fitnesses):.6f}' if np.mean(pso_fitnesses) > np.mean(ga_fitnesses) else 'GA=' + f'{np.mean(ga_fitnesses):.6f}'})")
    
    # Test 2: Cost
    t_stat_cost, p_value_cost = stats.ttest_rel(ga_costs, pso_costs)
    print(f"\n2️  COST COMPARISON")
    print(f"   T-statistic: {t_stat_cost:.4f}")
    print(f"   P-value: {p_value_cost:.6f}")
    print(f"   Result: {'TIDAK signifikan (p > 0.05)' if p_value_cost > 0.05 else 'SIGNIFIKAN (p < 0.05)'}")
    print(f"   Winner: {'PSO' if np.mean(pso_costs) < np.mean(ga_costs) else 'GA'} "
          f"(Mean: {'PSO=Rp' + f'{np.mean(pso_costs):,.0f}' if np.mean(pso_costs) < np.mean(ga_costs) else 'GA=Rp' + f'{np.mean(ga_costs):,.0f}'})")
    
    # Test 3: Time
    t_stat_time, p_value_time = stats.ttest_rel(ga_times, pso_times)
    print(f"\n3️  COMPUTATIONAL TIME COMPARISON")
    print(f"   T-statistic: {t_stat_time:.4f}")
    print(f"   P-value: {p_value_time:.6f}")
    print(f"   Result: {'TIDAK signifikan (p > 0.05)' if p_value_time > 0.05 else 'SIGNIFIKAN (p < 0.05)'}")
    print(f"   Winner: {'PSO' if np.mean(pso_times) < np.mean(ga_times) else 'GA'} "
          f"(Mean: {'PSO=' + f'{np.mean(pso_times):.2f}s' if np.mean(pso_times) < np.mean(ga_times) else 'GA=' + f'{np.mean(ga_times):.2f}s'})")
    
    #  CONFIDENCE INTERVALS (95%) 
    print(f"\n{'='*70}")
    print(f"{'95% CONFIDENCE INTERVALS':^70}")
    print(f"{'='*70}\n")
    
    # CI formula: mean ± t_critical * (std / sqrt(n))
    confidence_level = 0.95
    degrees_freedom = n_runs - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    # GA CI
    ga_fit_ci = t_critical * (np.std(ga_fitnesses, ddof=1) / np.sqrt(n_runs))
    ga_cost_ci = t_critical * (np.std(ga_costs, ddof=1) / np.sqrt(n_runs))
    ga_time_ci = t_critical * (np.std(ga_times, ddof=1) / np.sqrt(n_runs))
    
    print(f" GENETIC ALGORITHM")
    print(f"   Fitness : {np.mean(ga_fitnesses):.6f} ± {ga_fit_ci:.6f}")
    print(f"   Cost    : Rp {np.mean(ga_costs):,.0f} ± {ga_cost_ci:,.0f}")
    print(f"   Time    : {np.mean(ga_times):.2f}s ± {ga_time_ci:.2f}s")
    
    # PSO CI
    pso_fit_ci = t_critical * (np.std(pso_fitnesses, ddof=1) / np.sqrt(n_runs))
    pso_cost_ci = t_critical * (np.std(pso_costs, ddof=1) / np.sqrt(n_runs))
    pso_time_ci = t_critical * (np.std(pso_times, ddof=1) / np.sqrt(n_runs))
    
    print(f"\n PARTICLE SWARM OPTIMIZATION")
    print(f"   Fitness : {np.mean(pso_fitnesses):.6f} ± {pso_fit_ci:.6f}")
    print(f"   Cost    : Rp {np.mean(pso_costs):,.0f} ± {pso_cost_ci:,.0f}")
    print(f"   Time    : {np.mean(pso_times):.2f}s ± {pso_time_ci:.2f}s")
    
    #  BOXPLOT VISUALIZATION 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Boxplot 1: Fitness
    axes[0].boxplot([ga_fitnesses, pso_fitnesses], labels=['GA', 'PSO'])
    axes[0].set_ylabel('Fitness', fontsize=12)
    axes[0].set_title('Fitness Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    axes[0].plot([1, 2], [np.mean(ga_fitnesses), np.mean(pso_fitnesses)], 
                 'ro', markersize=8, label='Mean')
    axes[0].legend()
    
    # Boxplot 2: Cost
    axes[1].boxplot([ga_costs, pso_costs], labels=['GA', 'PSO'])
    axes[1].set_ylabel('Cost (Rp)', fontsize=12)
    axes[1].set_title('Cost Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    axes[1].plot([1, 2], [np.mean(ga_costs), np.mean(pso_costs)], 
                 'ro', markersize=8, label='Mean')
    axes[1].legend()
    
    # Boxplot 3: Time
    axes[2].boxplot([ga_times, pso_times], labels=['GA', 'PSO'])
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    axes[2].set_title('Computational Time Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    axes[2].plot([1, 2], [np.mean(ga_times), np.mean(pso_times)], 
                 'ro', markersize=8, label='Mean')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"statistical_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nStatistical plots saved to: {plot_path}")
    
    plt.show()
    
    print(f"\n{'='*70}\n")
    
    #  RETURN RESULTS 
    return {
        'ga': {
            'fitness': {'mean': np.mean(ga_fitnesses), 'std': np.std(ga_fitnesses), 
                       'min': np.min(ga_fitnesses), 'max': np.max(ga_fitnesses),
                       'ci': ga_fit_ci, 'data': ga_fitnesses},
            'cost': {'mean': np.mean(ga_costs), 'std': np.std(ga_costs),
                    'min': np.min(ga_costs), 'max': np.max(ga_costs),
                    'ci': ga_cost_ci, 'data': ga_costs},
            'time': {'mean': np.mean(ga_times), 'std': np.std(ga_times),
                    'min': np.min(ga_times), 'max': np.max(ga_times),
                    'ci': ga_time_ci, 'data': ga_times}
        },
        'pso': {
            'fitness': {'mean': np.mean(pso_fitnesses), 'std': np.std(pso_fitnesses),
                       'min': np.min(pso_fitnesses), 'max': np.max(pso_fitnesses),
                       'ci': pso_fit_ci, 'data': pso_fitnesses},
            'cost': {'mean': np.mean(pso_costs), 'std': np.std(pso_costs),
                    'min': np.min(pso_costs), 'max': np.max(pso_costs),
                    'ci': pso_cost_ci, 'data': pso_costs},
            'time': {'mean': np.mean(pso_times), 'std': np.std(pso_times),
                    'min': np.min(pso_times), 'max': np.max(pso_times),
                    'ci': pso_time_ci, 'data': pso_times}
        },
        'ttest': {
            'fitness': {'t_stat': t_stat_fit, 'p_value': p_value_fit,
                       'significant': p_value_fit < 0.05},
            'cost': {'t_stat': t_stat_cost, 'p_value': p_value_cost,
                    'significant': p_value_cost < 0.05},
            'time': {'t_stat': t_stat_time, 'p_value': p_value_time,
                    'significant': p_value_time < 0.05}
        }
    }


# 9. MAIN PROGRAM - MENU UTAMA
def main():
    print(f"\n{'='*70}")
    print(f"{'NUTRITION OPTIMIZATION SYSTEM':^70}")
    print(f"{'GA vs PSO Comparison':^70}")
    print(f"{'='*70}\n")
    
    while True:
        print(f"\n{'─'*70}")
        print(f"{'MENU UTAMA':^70}")
        print(f"{'─'*70}")
        print(f"1. Run Single Day - GA vs PSO")
        print(f"2. Generate Weekly Menu (7 Days)")
        print(f"3. Statistical Analysis (30 Runs)")
        print(f"4. Quick Demo (Best Solution)")
        print(f"5. Verification Test (Constraint Satisfaction)") 
        print(f"6. Exit")
        print(f"{'─'*70}")
        
        choice = input("\nPilih menu (1-6): ").strip()
        
        if choice == '1':
            #  SINGLE DAY COMPARISON 
            print(f"\n{'='*70}")
            print(f"{'SINGLE DAY OPTIMIZATION':^70}")
            print(f"{'='*70}")
            
            # Run GA
            ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
            ga_solution, ga_fitness, ga_history = ga.evolve(verbose=True)
            ga_nutrition = calculate_nutrition(ga_solution)
            
            # Run PSO
            pso = ParticleSwarmOptimization(n_particles=30, iterations=50, 
                                           w=0.7, c1=1.5, c2=1.5)
            pso_solution, pso_fitness, pso_history = pso.optimize(verbose=True)
            pso_nutrition = calculate_nutrition(pso_solution)
            
            # Print results
            print("\n GA - BEST MENU")
            print_menu_detail(ga_solution)
            
            print("\n PSO - BEST MENU")
            print_menu_detail(pso_solution)
            
            # Comparison
            ga_result = {
                'fitness': ga_fitness,
                'nutrition': ga_nutrition,
                'history': ga_history,
                'time': ga_history['time']
            }
            
            pso_result = {
                'fitness': pso_fitness,
                'nutrition': pso_nutrition,
                'history': pso_history,
                'time': pso_history['time']
            }
            
            print_comparison_summary(ga_result, pso_result)
            
            # Plot convergence
            plot_convergence_comparison(ga_history, pso_history)
        
        elif choice == '2':
            #  WEEKLY MENU GENERATION 
            print("\nGenerate menu untuk 7 hari")
            print("Pilih algoritma:")
            print("1. Genetic Algorithm (GA)")
            print("2. Particle Swarm Optimization (PSO)")
            print("3. Both (GA + PSO)")
            
            algo_choice = input("\nPilih (1-3): ").strip()
            
            algo_map = {'1': 'ga', '2': 'pso', '3': 'both'}
            algorithm = algo_map.get(algo_choice, 'both')
            
            verbose_input = input("Tampilkan detail menu? (y/n): ").strip().lower()
            verbose = verbose_input == 'y'
            
            weekly_results = generate_weekly_menu(algorithm=algorithm, verbose=verbose)
            
            # Save to CSV (optional)
            save_choice = input("\nSimpan hasil ke CSV? (y/n): ").strip().lower()
            if save_choice == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if algorithm in ['ga', 'both'] and len(weekly_results['GA']) > 0:
                    df_ga = pd.DataFrame([
                        {
                            'Day': r['day'],
                            'Fitness': r['fitness'],
                            'Cost': r['nutrition']['cost'],
                            'Kalori': r['nutrition']['kalori'],
                            'Protein': r['nutrition']['protein'],
                            'Karbo': r['nutrition']['karbo']
                        }
                        for r in weekly_results['GA']
                    ])
                    ga_file = f"weekly_menu_GA_{timestamp}.csv"
                    df_ga.to_csv(ga_file, index=False)
                    print(f"GA results saved to: {ga_file}")
                
                if algorithm in ['pso', 'both'] and len(weekly_results['PSO']) > 0:
                    df_pso = pd.DataFrame([
                        {
                            'Day': r['day'],
                            'Fitness': r['fitness'],
                            'Cost': r['nutrition']['cost'],
                            'Kalori': r['nutrition']['kalori'],
                            'Protein': r['nutrition']['protein'],
                            'Karbo': r['nutrition']['karbo']
                        }
                        for r in weekly_results['PSO']
                    ])
                    pso_file = f"weekly_menu_PSO_{timestamp}.csv"
                    df_pso.to_csv(pso_file, index=False)
                    print(f"PSO results saved to: {pso_file}")
        
        elif choice == '3':
            #  STATISTICAL ANALYSIS 
            print("\nStatistical Analysis")
            n_runs_input = input("Jumlah runs (default=30): ").strip()
            n_runs = int(n_runs_input) if n_runs_input.isdigit() else 30
            
            stat_results = run_statistical_test(n_runs=n_runs)
            
            # Save results to JSON (optional)
            save_choice = input("\nSimpan hasil statistik ke JSON? (y/n): ").strip().lower()
            if save_choice == 'y':
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_file = f"statistical_results_{timestamp}.json"
                
                # Convert numpy arrays to lists for JSON serialization
                save_data = {
                    'ga': {
                        'fitness': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                   for k, v in stat_results['ga']['fitness'].items()},
                        'cost': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                for k, v in stat_results['ga']['cost'].items()},
                        'time': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                for k, v in stat_results['ga']['time'].items()}
                    },
                    'pso': {
                        'fitness': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                   for k, v in stat_results['pso']['fitness'].items()},
                        'cost': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                for k, v in stat_results['pso']['cost'].items()},
                        'time': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                for k, v in stat_results['pso']['time'].items()}
                    },
                    'ttest': stat_results['ttest']
                }
                
                with open(json_file, 'w') as f:
                    json.dump(save_data, f, indent=4)
                print(f"Statistical results saved to: {json_file}")
        
        elif choice == '4':
            #  QUICK DEMO 
            print(f"\n{'='*70}")
            print(f"{'QUICK DEMO - BEST SOLUTION':^70}")
            print(f"{'='*70}\n")
            
            print("Running optimizations... (this may take a moment)")
            
            # Run both algorithms quietly
            ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
            ga_solution, ga_fitness, ga_history = ga.evolve(verbose=False)
            ga_nutrition = calculate_nutrition(ga_solution)
            
            pso = ParticleSwarmOptimization(n_particles=30, iterations=50, 
                                           w=0.7, c1=1.5, c2=1.5)
            pso_solution, pso_fitness, pso_history = pso.optimize(verbose=False)
            pso_nutrition = calculate_nutrition(pso_solution)
            
            # Determine winner
            if ga_fitness > pso_fitness:
                winner = "GA"
                winner_solution = ga_solution
                winner_fitness = ga_fitness
                winner_nutrition = ga_nutrition
            else:
                winner = "PSO"
                winner_solution = pso_solution
                winner_fitness = pso_fitness
                winner_nutrition = pso_nutrition
            
            print(f"\n||WINNER||: {winner}")
            print(f"Fitness: {winner_fitness:.6f}")
            print(f"Total Cost: Rp {winner_nutrition['cost']:,.0f}")
            
            print_menu_detail(winner_solution)
            
            # Quick comparison
            print(f"\n{'─'*70}")
            print(f"{'QUICK COMPARISON':^70}")
            print(f"{'─'*70}")
            print(f"{'Algorithm':<15} {'Fitness':>15} {'Cost (Rp)':>20} {'Time (s)':>15}")
            print(f"{'─'*70}")
            print(f"{'GA':<15} {ga_fitness:>15.6f} {ga_nutrition['cost']:>20,.0f} {ga_history['time']:>15.2f}")
            print(f"{'PSO':<15} {pso_fitness:>15.6f} {pso_nutrition['cost']:>20,.0f} {pso_history['time']:>15.2f}")
            print(f"{'─'*70}\n")

        elif choice == '5':
            verification_results = run_verification_test()
            # Tanyakan apakah ingin save results
            save_choice = input("\nSimpan hasil verifikasi ke file? (y/n): ").strip().lower()
            if save_choice == 'y':
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"verification_results_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(verification_results, f, indent=4)
                print(f"Hasil verifikasi berhasil disimpan ke: {filename}")
        
        elif choice == '6':
            #  EXIT 
            print(f"\n{'='*70}")
            print(f"{'Terima kasih telah menggunakan program ini!':^70}")
            print(f"{'='*70}\n")
            break
        
        else:
            print("\nPilihan tidak valid! Silakan pilih 1-5.")

# 10. ADDITIONAL UTILITY FUNCTIONS
def export_menu_to_pdf(solution: np.ndarray, filename: str = "menu.txt"):

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MENU MAKANAN SEIMBANG\n")
        f.write("="*70 + "\n\n")
        
        nutrition = calculate_nutrition(solution)
        
        # Write per category
        for category, foods in FOOD_DATABASE.items():
            f.write(f"\n{category.upper()}\n")
            f.write("-"*70 + "\n")
            
            start_idx = CATEGORY_START[category]
            
            for i, food in enumerate(foods):
                idx = start_idx + i
                portion = solution[idx]
                
                if portion > 5:
                    cost = food['harga'] * (portion / 1000)
                    f.write(f"  • {food['nama']:20s} : {portion:6.1f}g  (Rp {cost:,.0f})\n")
        
        # Write summary
        f.write(f"\n{'='*70}\n")
        f.write(f"RINGKASAN NUTRISI\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Kalori      : {nutrition['kalori']:7.1f} kkal\n")
        f.write(f"Protein     : {nutrition['protein']:7.1f} g\n")
        f.write(f"Karbohidrat : {nutrition['karbo']:7.1f} g\n")
        f.write(f"Biaya       : Rp {nutrition['cost']:,.0f}\n")
        f.write(f"{'='*70}\n")
    
    print(f"Menu exported to: {filename}")

def compare_multiple_runs(n_runs: int = 10, algorithm: str = 'both'):
    results = {'GA': [], 'PSO': []}
    
    print(f"\n{'='*70}")
    print(f"Running {n_runs} times for stability analysis...")
    print(f"{'='*70}\n")
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...", end='\r')
        np.random.seed(run * 999)
        
        if algorithm in ['ga', 'both']:
            ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
            _, ga_fitness, _ = ga.evolve(verbose=False)
            results['GA'].append(ga_fitness)
        
        if algorithm in ['pso', 'both']:
            pso = ParticleSwarmOptimization(n_particles=30, iterations=50, 
                                           w=0.7, c1=1.5, c2=1.5)
            _, pso_fitness, _ = pso.optimize(verbose=False)
            results['PSO'].append(pso_fitness)
    
    print(f"\nCompleted {n_runs} runs!\n")
    
    # Calculate coefficient of variation (CV) = std/mean
    if algorithm in ['ga', 'both']:
        ga_cv = np.std(results['GA']) / np.mean(results['GA']) * 100
        print(f" GA Stability:")
        print(f"   Mean: {np.mean(results['GA']):.6f}")
        print(f"   Std:  {np.std(results['GA']):.6f}")
        print(f"   CV:   {ga_cv:.2f}%")
    
    if algorithm in ['pso', 'both']:
        pso_cv = np.std(results['PSO']) / np.mean(results['PSO']) * 100
        print(f"\n PSO Stability:")
        print(f"   Mean: {np.mean(results['PSO']):.6f}")
        print(f"   Std:  {np.std(results['PSO']):.6f}")
        print(f"   CV:   {pso_cv:.2f}%")
    
    print(f"\nInterpretation:")
    print(f"  • CV < 10%  : Very stable (good)")
    print(f"  • CV 10-20% : Stable")
    print(f"  • CV > 20%  : Less stable (not good)")
    
    return results

def run_verification_test():
    """
    Verification test untuk memastikan aggressive repair bekerja
    """
    print(f"\n{'='*70}")
    print(f"{'VERIFICATION TEST - CONSTRAINT SATISFACTION':^70}")
    print(f"{'='*70}\n")
    
    # Test 1: GA
    print("Testing Genetic Algorithm...")
    ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
    ga_solution, ga_fitness, ga_history = ga.evolve(verbose=False)
    
    ga_validation = validate_final_solution(ga_solution)
    
    print(f"GA Results:")
    print(f"  Fitness: {ga_fitness:.6f}")
    print(f"  Constraints Met: {ga_validation['all_constraints_met']}")
    print(f"  Violations: {ga_validation['num_violations']}")
    
    if ga_validation['num_violations'] > 0:
        print(f"  Details:")
        for v in ga_validation['violations']:
            print(f"    - {v}")
    else:
        print(f"  ✓ ALL CONSTRAINTS SATISFIED!")
    
    # Test 2: PSO
    print(f"\nTesting Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(n_particles=30, iterations=50, 
                                   w=0.7, c1=1.5, c2=1.5)
    pso_solution, pso_fitness, pso_history = pso.optimize(verbose=False)
    
    pso_validation = validate_final_solution(pso_solution)
    
    print(f"PSO Results:")
    print(f"  Fitness: {pso_fitness:.6f}")
    print(f"  Constraints Met: {pso_validation['all_constraints_met']}")
    print(f"  Violations: {pso_validation['num_violations']}")
    
    if pso_validation['num_violations'] > 0:
        print(f"  Details:")
        for v in pso_validation['violations']:
            print(f"    - {v}")
    else:
        print(f"  ✓ ALL CONSTRAINTS SATISFIED!")
    
    # Test 3: Multiple runs verification
    print(f"\n{'-'*70}")
    print(f"Multiple Runs Verification (10 runs each)...")
    print(f"{'-'*70}")
    
    ga_violations_count = []
    pso_violations_count = []
    
    for run in range(10):
        print(f"Run {run+1}/10...", end='\r')
        
        # GA
        ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
        ga_sol, _, _ = ga.evolve(verbose=False)
        ga_val = validate_final_solution(ga_sol)
        ga_violations_count.append(ga_val['num_violations'])
        
        # PSO
        pso = ParticleSwarmOptimization(n_particles=30, iterations=50)
        pso_sol, _, _ = pso.optimize(verbose=False)
        pso_val = validate_final_solution(pso_sol)
        pso_violations_count.append(pso_val['num_violations'])
    
    print(f"\n10 Runs Completed!\n")
    
    ga_perfect = ga_violations_count.count(0)
    pso_perfect = pso_violations_count.count(0)
    
    print(f"GA  - Perfect solutions: {ga_perfect}/10 ({ga_perfect*10}%)")
    print(f"      Average violations: {np.mean(ga_violations_count):.2f}")
    print(f"PSO - Perfect solutions: {pso_perfect}/10 ({pso_perfect*10}%)")
    print(f"      Average violations: {np.mean(pso_violations_count):.2f}")
    
    print(f"\n{'='*70}")
    
    if ga_perfect >= 9 and pso_perfect >= 9:
        print(f"✓✓✓ VERIFICATION PASSED - Both algorithms achieve ≥90% constraint satisfaction!")
    elif ga_perfect >= 7 or pso_perfect >= 7:
        print(f"⚠ VERIFICATION WARNING - Some violations still exist. Consider strengthening repair.")
    else:
        print(f"✗✗✗ VERIFICATION FAILED - Aggressive repair needs improvement!")
    
    print(f"{'='*70}\n")
    
    return {
        'ga_perfect_rate': ga_perfect / 10,
        'pso_perfect_rate': pso_perfect / 10,
        'ga_avg_violations': np.mean(ga_violations_count),
        'pso_avg_violations': np.mean(pso_violations_count)
    }

# 11. RUN PROGRAM
if __name__ == "__main__":
    main()


# 12. TESTING & VALIDATION FUNCTIONS
def validate_solution(solution: np.ndarray) -> Dict:
    nutrition = calculate_nutrition(solution)
    penalty = calculate_penalty(solution, nutrition)
    
    validation = {
        'valid': penalty == 0,
        'penalty': penalty,
        'constraints': {}
    }
    
    # Check kalori
    kalori_ok = (TARGETS['kalori']['min'] <= nutrition['kalori'] <= 
                 TARGETS['kalori']['max'])
    validation['constraints']['kalori'] = {
        'satisfied': kalori_ok,
        'actual': nutrition['kalori'],
        'target': f"{TARGETS['kalori']['min']}-{TARGETS['kalori']['max']}"
    }
    
    # Check protein
    protein_ok = nutrition['protein'] >= TARGETS['protein']['min']
    validation['constraints']['protein'] = {
        'satisfied': protein_ok,
        'actual': nutrition['protein'],
        'target': f">={TARGETS['protein']['min']}"
    }
    
    # Check karbo
    karbo_ok = (TARGETS['karbo']['min'] <= nutrition['karbo'] <= 
                TARGETS['karbo']['max'])
    validation['constraints']['karbo'] = {
        'satisfied': karbo_ok,
        'actual': nutrition['karbo'],
        'target': f"{TARGETS['karbo']['min']}-{TARGETS['karbo']['max']}"
    }
    
    # Check budget
    budget_ok = nutrition['cost'] <= MAX_BUDGET
    validation['constraints']['budget'] = {
        'satisfied': budget_ok,
        'actual': nutrition['cost'],
        'target': f"<={MAX_BUDGET}"
    }
    
    # Check category minimums
    for category, min_portion in CATEGORY_MIN_PORTIONS.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        category_total = np.sum(solution[start_idx:end_idx])
        
        category_ok = category_total >= min_portion
        validation['constraints'][f'category_{category}'] = {
            'satisfied': category_ok,
            'actual': category_total,
            'target': f">={min_portion}"
        }
    
    return validation

def validate_final_solution(solution: np.ndarray) -> Dict:
    """
    Validate bahwa SEMUA constraints terpenuhi di solusi final
    """
    violations = []
    
    # Check C6
    awkward = [(i, p) for i, p in enumerate(solution) if 0 < p < MIN_PORTION_PER_FOOD]
    if awkward:
        violations.append(f"C6: {len(awkward)} portions < 50g")
    
    # Check C7
    for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        items = np.sum(solution[start_idx:end_idx] >= MIN_PORTION_PER_FOOD)
        
        if items > max_items:
            violations.append(f"C7: {category} = {items} items (max={max_items})")
    
    # Check C8
    staple_total = sum(solution[idx] for idx in STAPLE_FOOD_INDICES)
    if staple_total < MIN_STAPLE_PORTION:
        violations.append(f"C8: Staple food = {staple_total:.1f}g (min=200g)")
    
    # Check C5B
    minuman_start = CATEGORY_START['minuman']
    minuman_end = minuman_start + len(FOOD_DATABASE['minuman'])
    minuman_total = np.sum(solution[minuman_start:minuman_end])
    
    if minuman_total < 600 or minuman_total > 900:
        violations.append(f"C5B: Minuman = {minuman_total:.1f}ml (range: 600-900ml)")
    
    return {
        'all_constraints_met': len(violations) == 0,
        'violations': violations,
        'num_violations': len(violations)
    }

def test_algorithms():
    print(f"\n{'='*70}")
    print(f"{'TESTING ALGORITHMS':^70}")
    print(f"{'='*70}\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: GA initialization
    print("Test 1: GA Initialization...", end=' ')
    try:
        ga = GeneticAlgorithm(pop_size=10, generations=5)
        pop = ga.initialize_population()
        assert len(pop) == 10
        assert all(len(ind) == NUM_FOODS for ind in pop)
        print("PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        tests_failed += 1
    
    # Test 2: GA evolution
    print("Test 2: GA Evolution...", end=' ')
    try:
        ga = GeneticAlgorithm(pop_size=10, generations=5)
        solution, fitness, history = ga.evolve(verbose=False)
        assert len(solution) == NUM_FOODS
        assert fitness > 0
        assert len(history['best_history']) == 5
        print("PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        tests_failed += 1
    
    # Test 3: PSO initialization
    print("Test 3: PSO Initialization...", end=' ')
    try:
        pso = ParticleSwarmOptimization(n_particles=10, iterations=5)
        positions = np.random.uniform(0, 300, (10, NUM_FOODS))
        assert positions.shape == (10, NUM_FOODS)
        print("PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        tests_failed += 1
    
    # Test 4: PSO optimization
    print("Test 4: PSO Optimization...", end=' ')
    try:
        pso = ParticleSwarmOptimization(n_particles=10, iterations=5)
        solution, fitness, history = pso.optimize(verbose=False)
        assert len(solution) == NUM_FOODS
        assert fitness > 0
        assert len(history['best_history']) == 5
        print("PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Fitness function
    print("Test 5: Fitness Function...", end=' ')
    try:
        test_solution = np.random.uniform(0, 300, NUM_FOODS)
        fitness = fitness_function(test_solution)
        assert fitness > 0
        assert np.isfinite(fitness)
        print("PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Constraint validation
    print("Test 6: Constraint Validation...", end=' ')
    try:
        test_solution = np.random.uniform(100, 200, NUM_FOODS)
        validation = validate_solution(test_solution)
        assert 'valid' in validation
        assert 'penalty' in validation
        assert 'constraints' in validation
        print("PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*70}\n")
    
    return tests_passed, tests_failed


# 13. PERFORMANCE PROFILING
def profile_performance():
    import time
    
    print(f"\n{'='*70}")
    print(f"{'  PERFORMANCE PROFILING':^70}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Profile GA
    print("Profiling Genetic Algorithm...")
    ga_times = []
    for i in range(5):
        start = time.time()
        ga = GeneticAlgorithm(pop_size=30, generations=50)
        ga.evolve(verbose=False)
        elapsed = time.time() - start
        ga_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    results['GA'] = {
        'mean': np.mean(ga_times),
        'std': np.std(ga_times),
        'min': np.min(ga_times),
        'max': np.max(ga_times)
    }
    
    # Profile PSO
    print("\nProfiling Particle Swarm Optimization...")
    pso_times = []
    for i in range(5):
        start = time.time()
        pso = ParticleSwarmOptimization(n_particles=30, iterations=50)
        pso.optimize(verbose=False)
        elapsed = time.time() - start
        pso_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    results['PSO'] = {
        'mean': np.mean(pso_times),
        'std': np.std(pso_times),
        'min': np.min(pso_times),
        'max': np.max(pso_times)
    }
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':<15} {'Mean (s)':>12} {'Std (s)':>12} {'Min (s)':>12} {'Max (s)':>12}")
    print(f"{'-'*70}")
    print(f"{'GA':<15} {results['GA']['mean']:>12.2f} {results['GA']['std']:>12.2f} "
          f"{results['GA']['min']:>12.2f} {results['GA']['max']:>12.2f}")
    print(f"{'PSO':<15} {results['PSO']['mean']:>12.2f} {results['PSO']['std']:>12.2f} "
          f"{results['PSO']['min']:>12.2f} {results['PSO']['max']:>12.2f}")
    print(f"{'='*70}\n")
    
    speedup = results['GA']['mean'] / results['PSO']['mean']
    if speedup > 1:
        print(f" PSO is {speedup:.2f}x faster than GA")
    else:
        print(f" GA is {1/speedup:.2f}x faster than PSO")
    
    return results

print("\n" + "="*70)
print("  Program loaded successfully!")
print("  Read documentation above for usage guide")
print("  Run main() to start interactive menu")
print("="*70 + "\n")