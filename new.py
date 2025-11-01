# ============================================================================
# NUTRITION OPTIMIZATION USING GA & PSO - COMPLETE VERSION
# Project: Optimasi Menu Makanan Seimbang (4 Sehat 5 Sempurna)
# Author: Mochamad Faisal Akbar (L0122094)
# Course: Kecerdasan Komputasional
# Features: 
#   - GA vs PSO comparison
#   - 7-day menu generation
#   - Statistical analysis (t-test over 30 runs)
#   - Comprehensive reporting
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict
import time
import os
from datetime import datetime

# ============================================================================
# 1. DATASET - 50 Makanan Indonesia (10 per kategori)
# Sumber: Tabel Komposisi Pangan Indonesia (TKPI) 2017
# Harga: Estimasi pasar Surakarta 2025
# ============================================================================

FOOD_DATABASE = {
    'buah': [
        {'nama': 'Pisang', 'kalori': 89, 'protein': 1.1, 'karbo': 22.8, 'harga': 5000},
        {'nama': 'Apel', 'kalori': 52, 'protein': 0.3, 'karbo': 14, 'harga': 12000},
        {'nama': 'Jeruk', 'kalori': 47, 'protein': 0.9, 'karbo': 12, 'harga': 8000},
        {'nama': 'Mangga', 'kalori': 60, 'protein': 0.8, 'karbo': 15, 'harga': 10000},
        {'nama': 'Pepaya', 'kalori': 43, 'protein': 0.5, 'karbo': 11, 'harga': 6000},
        {'nama': 'Semangka', 'kalori': 30, 'protein': 0.6, 'karbo': 8, 'harga': 5000},
        {'nama': 'Anggur', 'kalori': 69, 'protein': 0.7, 'karbo': 18, 'harga': 15000},
        {'nama': 'Melon', 'kalori': 34, 'protein': 0.8, 'karbo': 8, 'harga': 7000},
        {'nama': 'Pir', 'kalori': 57, 'protein': 0.4, 'karbo': 15, 'harga': 13000},
        {'nama': 'Nanas', 'kalori': 50, 'protein': 0.5, 'karbo': 13, 'harga': 6000},
    ],
    'karbohidrat': [
        {'nama': 'Nasi Putih', 'kalori': 130, 'protein': 2.7, 'karbo': 28, 'harga': 12000},
        {'nama': 'Roti Tawar', 'kalori': 265, 'protein': 9, 'karbo': 49, 'harga': 15000},
        {'nama': 'Mie Instant', 'kalori': 188, 'protein': 4.5, 'karbo': 27, 'harga': 3000},
        {'nama': 'Kentang', 'kalori': 77, 'protein': 2, 'karbo': 17, 'harga': 8000},
        {'nama': 'Singkong', 'kalori': 160, 'protein': 1.4, 'karbo': 38, 'harga': 5000},
        {'nama': 'Jagung', 'kalori': 86, 'protein': 3.3, 'karbo': 19, 'harga': 6000},
        {'nama': 'Ubi', 'kalori': 86, 'protein': 1.6, 'karbo': 20, 'harga': 7000},
        {'nama': 'Pasta', 'kalori': 158, 'protein': 5.8, 'karbo': 31, 'harga': 18000},
        {'nama': 'Oatmeal', 'kalori': 68, 'protein': 2.4, 'karbo': 12, 'harga': 25000},
        {'nama': 'Roti Gandum', 'kalori': 247, 'protein': 13, 'karbo': 41, 'harga': 20000},
    ],
    'protein': [
        {'nama': 'Ayam', 'kalori': 165, 'protein': 31, 'karbo': 0, 'harga': 35000},
        {'nama': 'Telur', 'kalori': 155, 'protein': 13, 'karbo': 1.1, 'harga': 20000},
        {'nama': 'Tempe', 'kalori': 193, 'protein': 19, 'karbo': 9, 'harga': 8000},
        {'nama': 'Tahu', 'kalori': 76, 'protein': 8, 'karbo': 1.9, 'harga': 6000},
        {'nama': 'Ikan Lele', 'kalori': 168, 'protein': 26, 'karbo': 0, 'harga': 25000},
        {'nama': 'Daging Sapi', 'kalori': 250, 'protein': 26, 'karbo': 0, 'harga': 120000},
        {'nama': 'Ikan Tongkol', 'kalori': 144, 'protein': 23, 'karbo': 0, 'harga': 30000},
        {'nama': 'Udang', 'kalori': 99, 'protein': 24, 'karbo': 0.2, 'harga': 80000},
        {'nama': 'Kacang Merah', 'kalori': 127, 'protein': 8.7, 'karbo': 23, 'harga': 15000},
        {'nama': 'Kacang Hijau', 'kalori': 347, 'protein': 24, 'karbo': 63, 'harga': 18000},
    ],
    'sayur': [
        {'nama': 'Bayam', 'kalori': 23, 'protein': 2.9, 'karbo': 3.6, 'harga': 5000},
        {'nama': 'Kangkung', 'kalori': 19, 'protein': 3, 'karbo': 3, 'harga': 4000},
        {'nama': 'Wortel', 'kalori': 41, 'protein': 0.9, 'karbo': 10, 'harga': 8000},
        {'nama': 'Brokoli', 'kalori': 34, 'protein': 2.8, 'karbo': 7, 'harga': 15000},
        {'nama': 'Kol', 'kalori': 25, 'protein': 1.3, 'karbo': 6, 'harga': 6000},
        {'nama': 'Tomat', 'kalori': 18, 'protein': 0.9, 'karbo': 3.9, 'harga': 10000},
        {'nama': 'Timun', 'kalori': 15, 'protein': 0.7, 'karbo': 3.6, 'harga': 5000},
        {'nama': 'Terong', 'kalori': 25, 'protein': 1, 'karbo': 6, 'harga': 7000},
        {'nama': 'Buncis', 'kalori': 31, 'protein': 1.8, 'karbo': 7, 'harga': 9000},
        {'nama': 'Labu Siam', 'kalori': 19, 'protein': 0.8, 'karbo': 4.5, 'harga': 5000},
    ],
    'minuman': [
        {'nama': 'Susu Sapi', 'kalori': 61, 'protein': 3.2, 'karbo': 4.8, 'harga': 18000},
        {'nama': 'Teh Manis', 'kalori': 30, 'protein': 0, 'karbo': 8, 'harga': 1000},
        {'nama': 'Jus Jeruk', 'kalori': 45, 'protein': 0.7, 'karbo': 10, 'harga': 5000},
        {'nama': 'Air Kelapa', 'kalori': 19, 'protein': 0.7, 'karbo': 3.7, 'harga': 5000},
        {'nama': 'Susu Kedelai', 'kalori': 54, 'protein': 3.3, 'karbo': 6, 'harga': 8000},
        {'nama': 'Yogurt', 'kalori': 59, 'protein': 3.5, 'karbo': 4.7, 'harga': 12000},
        {'nama': 'Kopi Susu', 'kalori': 38, 'protein': 2, 'karbo': 5, 'harga': 3000},
        {'nama': 'Jus Alpukat', 'kalori': 160, 'protein': 2, 'karbo': 8.5, 'harga': 10000},
        {'nama': 'Air Putih', 'kalori': 0, 'protein': 0, 'karbo': 0, 'harga': 0},
        {'nama': 'Jus Tomat', 'kalori': 17, 'protein': 0.8, 'karbo': 3.9, 'harga': 6000},
    ]
}

# Flatten database untuk akses mudah
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
print(f"✅ Total makanan dalam database: {NUM_FOODS}")

# ============================================================================
# 2. NUTRITIONAL TARGETS (Angka Kecukupan Gizi Indonesia)
# Sumber: Peraturan Menteri Kesehatan RI No. 28 Tahun 2019
# ============================================================================

TARGETS = {
    'kalori': {'min': 1800, 'max': 2200, 'ideal': 2000},
    'protein': {'min': 50, 'max': 80, 'ideal': 60},
    'karbo': {'min': 250, 'max': 350, 'ideal': 300}
}

# Minimum portions per category (4 Sehat 5 Sempurna)
CATEGORY_MIN_PORTIONS = {
    'buah': 150,        # gram (2 porsi @ 75g)
    'karbohidrat': 300, # gram (3 porsi @ 100g)
    'protein': 150,     # gram (2 porsi @ 75g)
    'sayur': 200,       # gram (2-3 porsi @ 70-100g)
    'minuman': 200      # ml (1-2 gelas)
}

MAX_BUDGET = 50000  # Rp per hari

# ============================================================================
# 3. FITNESS FUNCTION & CONSTRAINTS
# ============================================================================

def calculate_nutrition(portions: np.ndarray) -> Dict:
    """
    Hitung total nutrisi dari porsi makanan
    
    Input: portions = array 50 elemen (gram)
    Output: {'kalori', 'protein', 'karbo', 'cost'}
    
    Formula:
      Total_Nutrient = Σ(i=0..49) [nutrient_i × (x_i / 100)]
      Total_Cost = Σ(i=0..49) [harga_i × (x_i / 1000)]
    """
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
    """
    Hitung penalty untuk constraint violations
    
    Constraints:
    C1: 1800 ≤ Kalori ≤ 2200
    C2: Protein ≥ 50g
    C3: 250 ≤ Karbo ≤ 350
    C4: Cost ≤ 50,000
    C5: Setiap kategori ≥ minimum portion
    
    Penalty menggunakan quadratic function:
      P = weight × (violation)²
    
    Returns: Total penalty (float)
    """
    penalty = 0
    
    # === C1: KALORI CONSTRAINTS ===
    # Penalty jika kalori < 1800 atau > 2200
    if nutrition['kalori'] < TARGETS['kalori']['min']:
        violation = TARGETS['kalori']['min'] - nutrition['kalori']
        penalty += violation ** 2 * 0.1  # weight = 0.1
    
    if nutrition['kalori'] > TARGETS['kalori']['max']:
        violation = nutrition['kalori'] - TARGETS['kalori']['max']
        penalty += violation ** 2 * 0.1
    
    # === C2: PROTEIN CONSTRAINT ===
    # Penalty jika protein < 50g
    if nutrition['protein'] < TARGETS['protein']['min']:
        violation = TARGETS['protein']['min'] - nutrition['protein']
        penalty += violation ** 2 * 2  # weight = 2.0 (penting!)
    
    # === C3: KARBOHIDRAT CONSTRAINTS ===
    # Penalty jika karbo < 250 atau > 350
    if nutrition['karbo'] < TARGETS['karbo']['min']:
        violation = TARGETS['karbo']['min'] - nutrition['karbo']
        penalty += violation ** 2 * 0.05
    
    if nutrition['karbo'] > TARGETS['karbo']['max']:
        violation = nutrition['karbo'] - TARGETS['karbo']['max']
        penalty += violation ** 2 * 0.05
    
    # === C4: BUDGET CONSTRAINT ===
    # Penalty jika cost > 50,000
    if nutrition['cost'] > MAX_BUDGET:
        violation = nutrition['cost'] - MAX_BUDGET
        penalty += violation ** 2 * 0.01
    
    # === C5: CATEGORY MINIMUM PORTIONS (4 Sehat 5 Sempurna) ===
    # Penalty jika total porsi kategori < minimum
    for category, min_portion in CATEGORY_MIN_PORTIONS.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        category_total = np.sum(portions[start_idx:end_idx])
        
        if category_total < min_portion:
            violation = min_portion - category_total
            penalty += violation ** 2 * 0.5
    
    return penalty

def fitness_function(portions: np.ndarray) -> float:
    """
    Main fitness function (higher is better)
    
    Objective: Minimize cost
    Formula:
      Fitness = 1000 / (Cost + Penalty + 1)
    
    Explanation:
    - Lower cost → Higher fitness ✅
    - Higher penalty → Lower fitness ❌
    - +1 to avoid division by zero
    - 1000 = scaling factor
    
    Returns: fitness value (float)
    """
    nutrition = calculate_nutrition(portions)
    penalty = calculate_penalty(portions, nutrition)
    cost = nutrition['cost']
    
    # Fitness formula
    fitness = 1000 / (cost + penalty + 1)
    
    return fitness

# ============================================================================
# 4. GENETIC ALGORITHM (GA)
# ============================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm for Menu Optimization
    
    Parameters:
    - pop_size: Population size (jumlah individu)
    - generations: Number of generations (iterasi)
    - pc: Crossover rate (probabilitas crossover)
    - pm: Mutation rate (probabilitas mutasi)
    
    Operators:
    - Selection: Tournament selection (k=3)
    - Crossover: Single-point crossover
    - Mutation: Gaussian mutation
    - Elitism: Keep top 10%
    """
    
    def __init__(self, pop_size=30, generations=50, pc=0.8, pm=0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc  # crossover rate
        self.pm = pm  # mutation rate
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self):
        """
        Create initial population randomly
        
        Returns: List of individuals (each = array 50 elemen)
        """
        population = []
        for _ in range(self.pop_size):
            # Random portions 0-300g per food
            individual = np.random.uniform(0, 300, NUM_FOODS)
            population.append(individual)
        return population
    
    def tournament_selection(self, population, fitnesses, k=3):
        """
        Tournament selection
        
        Process:
        1. Pilih k individu random
        2. Return yang fitness-nya tertinggi
        
        Args:
            population: List of individuals
            fitnesses: List of fitness values
            k: Tournament size (default=3)
        
        Returns: Selected individual (array)
        """
        # Pilih k kandidat secara random
        candidates_idx = np.random.choice(len(population), k, replace=False)
        candidates_fitness = [fitnesses[i] for i in candidates_idx]
        
        # Pilih yang fitness tertinggi
        winner_idx = candidates_idx[np.argmax(candidates_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """
        Single-point crossover
        
        Process:
        1. Pilih random crossover point
        2. Tukar segmen setelah crossover point
        
        Example:
          parent1 = [a1, a2, | a3, a4, a5]  (point=2)
          parent2 = [b1, b2, | b3, b4, b5]
          →
          child1  = [a1, a2, | b3, b4, b5]
          child2  = [b1, b2, | a3, a4, a5]
        
        Returns: child1, child2
        """
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
        """
        Gaussian mutation
        
        Process:
        1. Untuk setiap gene (makanan)
        2. Dengan probabilitas pm:
           - Tambah Gaussian noise (mean=0, std=30)
           - Clip ke range [0, 500]
        
        Args:
            individual: Array 50 elemen
        
        Returns: Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(NUM_FOODS):
            if np.random.rand() < self.pm:
                # Add Gaussian noise
                noise = np.random.normal(0, 30)  # std=30
                mutated[i] += noise
                # Clip to valid range
                mutated[i] = np.clip(mutated[i], 0, 500)
        
        return mutated
    
    def evolve(self, verbose=True):
        """
        Main GA evolution loop
        
        Process:
        1. Initialize population
        2. For each generation:
           a. Evaluate fitness
           b. Selection
           c. Crossover
           d. Mutation
           e. Elitism (keep best 10%)
        3. Return best solution
        
        Returns: best_solution, best_fitness, history
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧬 GENETIC ALGORITHM")
            print(f"{'='*60}")
            print(f"Population: {self.pop_size} | Generations: {self.generations}")
            print(f"Crossover Rate: {self.pc} | Mutation Rate: {self.pm}")
            print(f"{'='*60}\n")
        
        # Initialize
        population = self.initialize_population()
        best_solution = None
        best_fitness = -np.inf
        
        start_time = time.time()
        
        # Evolution loop
        for gen in range(self.generations):
            # === EVALUATE FITNESS ===
            fitnesses = [fitness_function(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
            
            # Track history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            
            # Print progress
            if verbose and (gen % 10 == 0 or gen == self.generations - 1):
                print(f"Gen {gen:3d} | Best Fitness: {best_fitness:.6f} | "
                      f"Avg: {np.mean(fitnesses):.6f}")
            
            # === SELECTION ===
            # Elitism: Keep top 10%
            elite_count = max(2, int(0.1 * self.pop_size))
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            elites = [population[i].copy() for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            # Fill rest with offspring
            while len(new_population) < self.pop_size:
                # === TOURNAMENT SELECTION ===
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # === CROSSOVER ===
                child1, child2 = self.crossover(parent1, parent2)
                
                # === MUTATION ===
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Update population
            population = new_population[:self.pop_size]
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ GA Completed in {elapsed_time:.2f} seconds")
            print(f"Best Fitness: {best_fitness:.6f}")
            print(f"{'='*60}\n")
        
        return best_solution, best_fitness, {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history,
            'time': elapsed_time
        }

# ============================================================================
# 5. PARTICLE SWARM OPTIMIZATION (PSO)
# ============================================================================

class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for Menu Optimization
    
    Parameters:
    - n_particles: Number of particles (swarm size)
    - iterations: Number of iterations
    - w: Inertia weight
    - c1: Cognitive coefficient (personal best)
    - c2: Social coefficient (global best)
    
    Update equations:
      v(t+1) = w·v(t) + c1·r1·(pbest - x) + c2·r2·(gbest - x)
      x(t+1) = x(t) + v(t+1)
    """
    
    def __init__(self, n_particles=30, iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w    # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def optimize(self, verbose=True):
        """
        Main PSO optimization loop
        
        Process:
        1. Initialize particles (position & velocity)
        2. For each iteration:
           a. Evaluate fitness
           b. Update personal best
           c. Update global best
           d. Update velocity
           e. Update position
        3. Return best solution
        
        Returns: best_solution, best_fitness, history
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🌊 PARTICLE SWARM OPTIMIZATION")
            print(f"{'='*60}")
            print(f"Particles: {self.n_particles} | Iterations: {self.iterations}")
            print(f"w: {self.w} | c1: {self.c1} | c2: {self.c2}")
            print(f"{'='*60}\n")
        
        # === INITIALIZE SWARM ===
        # Position: Random 0-300g per food
        positions = np.random.uniform(0, 300, (self.n_particles, NUM_FOODS))
        # Velocity: Random -50 to 50
        velocities = np.random.uniform(-50, 50, (self.n_particles, NUM_FOODS))
        
        # Personal best
        pbest_positions = positions.copy()
        pbest_fitness = np.array([fitness_function(p) for p in positions])
        
        # Global best
        gbest_idx = np.argmax(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        start_time = time.time()
        
        # === OPTIMIZATION LOOP ===
        for iter in range(self.iterations):
            for i in range(self.n_particles):
                # === EVALUATE FITNESS ===
                fitness = fitness_function(positions[i])
                
                # === UPDATE PERSONAL BEST ===
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                
                # === UPDATE GLOBAL BEST ===
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()
                
                # === UPDATE VELOCITY ===
                r1 = np.random.rand(NUM_FOODS)  # Random for cognitive
                r2 = np.random.rand(NUM_FOODS)  # Random for social
                
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                
                velocities[i] = (self.w * velocities[i] + cognitive + social)
                
                # === UPDATE POSITION ===
                positions[i] = positions[i] + velocities[i]
                
                # Clip to valid range [0, 500]
                positions[i] = np.clip(positions[i], 0, 500)
            
            # Track history
            self.best_fitness_history.append(gbest_fitness)
            self.avg_fitness_history.append(np.mean(pbest_fitness))
            
            # Print progress
            if verbose and (iter % 10 == 0 or iter == self.iterations - 1):
                print(f"Iter {iter:3d} | Best Fitness: {gbest_fitness:.6f} | "
                      f"Avg: {np.mean(pbest_fitness):.6f}")
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ PSO Completed in {elapsed_time:.2f} seconds")
            print(f"Best Fitness: {gbest_fitness:.6f}")
            print(f"{'='*60}\n")
        
        return gbest_position, gbest_fitness, {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history,
            'time': elapsed_time
        }

# ============================================================================
# 6. HELPER FUNCTIONS - REPORTING & VISUALIZATION
# ============================================================================

def print_menu_detail(solution: np.ndarray, day_number: int = None):
    """
    Print detailed menu dengan format yang rapi
    
    Args:
        solution: Array 50 elemen (porsi makanan)
        day_number: Nomor hari (opsional)
    """
    header = f"📋 MENU DETAIL" + (f" - HARI {day_number}" if day_number else "")
    print(f"\n{'='*70}")
    print(f"{header:^70}")
    print(f"{'='*70}")
    
    nutrition = calculate_nutrition(solution)
    
    # Print per kategori
    for category, foods in FOOD_DATABASE.items():
        print(f"\n🍴 {category.upper()}")
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
    print(f"📊 RINGKASAN NUTRISI")
    print(f"{'-'*70}")
    print(f"  Kalori      : {nutrition['kalori']:7.1f} kkal "
          f"({'✅' if TARGETS['kalori']['min'] <= nutrition['kalori'] <= TARGETS['kalori']['max'] else '❌'} "
          f"Target: {TARGETS['kalori']['min']}-{TARGETS['kalori']['max']})")
    print(f"  Protein     : {nutrition['protein']:7.1f} g    "
          f"({'✅' if nutrition['protein'] >= TARGETS['protein']['min'] else '❌'} "
          f"Target: ≥{TARGETS['protein']['min']})")
    print(f"  Karbohidrat : {nutrition['karbo']:7.1f} g    "
          f"({'✅' if TARGETS['karbo']['min'] <= nutrition['karbo'] <= TARGETS['karbo']['max'] else '❌'} "
          f"Target: {TARGETS['karbo']['min']}-{TARGETS['karbo']['max']})")
    print(f"  Biaya       : Rp {nutrition['cost']:,.0f}  "
          f"({'✅' if nutrition['cost'] <= MAX_BUDGET else '❌'} "
          f"Target: ≤Rp {MAX_BUDGET:,})")
    print(f"{'='*70}\n")

def plot_convergence_comparison(ga_history: Dict, pso_history: Dict, 
                                save_path: str = None):
    """
    Plot perbandingan konvergensi GA vs PSO
    
    Args:
        ga_history: Dictionary hasil GA
        pso_history: Dictionary hasil PSO
        save_path: Path untuk save plot (opsional)
    """
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
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()

def print_comparison_summary(ga_result: Dict, pso_result: Dict):
    """
    Print summary perbandingan GA vs PSO
    
    Args:
        ga_result: Dictionary hasil GA
        pso_result: Dictionary hasil PSO
    """
    print(f"\n{'='*70}")
    print(f"{'📊 PERBANDINGAN GA vs PSO':^70}")
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

# ============================================================================
# 7. MENU 7 HARI - RUN MULTIPLE TIMES
# ============================================================================

def generate_weekly_menu(algorithm='both', verbose=True):
    """
    Generate menu untuk 7 hari menggunakan GA dan/atau PSO
    
    Strategy: Run algoritma 7 kali dengan random seed berbeda
    
    Args:
        algorithm: 'ga', 'pso', atau 'both'
        verbose: Print detail menu atau tidak
    
    Returns: Dictionary hasil 7 hari
    """
    results = {'GA': [], 'PSO': []}
    days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    
    print(f"\n{'='*70}")
    print(f"{'🗓️  GENERATE MENU 7 HARI':^70}")
    print(f"{'='*70}\n")
    
    for day_num, day_name in enumerate(days, 1):
        print(f"\n{'~'*70}")
        print(f"{'HARI ' + str(day_num) + ': ' + day_name:^70}")
        print(f"{'~'*70}")
        
        # Set random seed untuk variasi
        np.random.seed(day_num * 42)
        
        # === RUN GA ===
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
                print(f"🧬 GA Fitness: {ga_fitness:.6f} | Cost: Rp {ga_nutrition['cost']:,.0f}")
        
        # === RUN PSO ===
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
                print(f"🌊 PSO Fitness: {pso_fitness:.6f} | Cost: Rp {pso_nutrition['cost']:,.0f}")
    
    # === SUMMARY 7 HARI ===
    print(f"\n{'='*70}")
    print(f"{'📊 SUMMARY 7 HARI':^70}")
    print(f"{'='*70}\n")
    
    if 'GA' in results and len(results['GA']) > 0:
        ga_costs = [r['nutrition']['cost'] for r in results['GA']]
        print(f"🧬 GA - Total Cost 7 Hari: Rp {sum(ga_costs):,.0f}")
        print(f"   Average per hari: Rp {np.mean(ga_costs):,.0f} ± {np.std(ga_costs):,.0f}")
    
    if 'PSO' in results and len(results['PSO']) > 0:
        pso_costs = [r['nutrition']['cost'] for r in results['PSO']]
        print(f"🌊 PSO - Total Cost 7 Hari: Rp {sum(pso_costs):,.0f}")
        print(f"   Average per hari: Rp {np.mean(pso_costs):,.0f} ± {np.std(pso_costs):,.0f}")
    
    print(f"{'='*70}\n")
    
    return results

# ============================================================================
# 8. STATISTICAL ANALYSIS - 30 RUNS dengan T-TEST
# ============================================================================

def run_statistical_test(n_runs=30):
    """
    Run GA dan PSO sebanyak n_runs kali untuk analisis statistik
    
    Includes:
    - Descriptive statistics (mean, std, min, max)
    - Paired t-test
    - Confidence intervals
    - Boxplot visualization
    
    Args:
        n_runs: Jumlah run (default=30)
    
    Returns: Dictionary hasil statistik
    """
    print(f"\n{'='*70}")
    print(f"{'📈 STATISTICAL ANALYSIS - ' + str(n_runs) + ' RUNS':^70}")
    print(f"{'='*70}\n")
    
    ga_fitnesses = []
    ga_costs = []
    ga_times = []
    
    pso_fitnesses = []
    pso_costs = []
    pso_times = []
    
    # === RUN 30 TIMES ===
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
    
    print(f"\n✅ Completed {n_runs} runs!\n")
    
    # === DESCRIPTIVE STATISTICS ===
    print(f"{'='*70}")
    print(f"{'📊 DESCRIPTIVE STATISTICS':^70}")
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
    print(f"\n🧬 GENETIC ALGORITHM")
    print(f"{'Fitness':<20} {np.mean(ga_fitnesses):>12.6f} {np.std(ga_fitnesses):>12.6f} "
          f"{np.min(ga_fitnesses):>12.6f} {np.max(ga_fitnesses):>12.6f}")
    print(f"{'Cost (Rp)':<20} {np.mean(ga_costs):>12,.0f} {np.std(ga_costs):>12,.0f} "
          f"{np.min(ga_costs):>12,.0f} {np.max(ga_costs):>12,.0f}")
    print(f"{'Time (s)':<20} {np.mean(ga_times):>12.2f} {np.std(ga_times):>12.2f} "
          f"{np.min(ga_times):>12.2f} {np.max(ga_times):>12.2f}")
    
    # PSO Statistics
    print(f"\n🌊 PARTICLE SWARM OPTIMIZATION")
    print(f"{'Fitness':<20} {np.mean(pso_fitnesses):>12.6f} {np.std(pso_fitnesses):>12.6f} "
          f"{np.min(pso_fitnesses):>12.6f} {np.max(pso_fitnesses):>12.6f}")
    print(f"{'Cost (Rp)':<20} {np.mean(pso_costs):>12,.0f} {np.std(pso_costs):>12,.0f} "
          f"{np.min(pso_costs):>12,.0f} {np.max(pso_costs):>12,.0f}")
    print(f"{'Time (s)':<20} {np.mean(pso_times):>12.2f} {np.std(pso_times):>12.2f} "
          f"{np.min(pso_times):>12.2f} {np.max(pso_times):>12.2f}")
    
    # === PAIRED T-TEST ===
    print(f"\n{'='*70}")
    print(f"{'📊 PAIRED T-TEST RESULTS':^70}")
    print(f"{'='*70}\n")
    
    # H0: μ_GA = μ_PSO (tidak ada perbedaan signifikan)
    # H1: μ_GA ≠ μ_PSO (ada perbedaan signifikan)
    # α = 0.05
    
    # Test 1: Fitness
    t_stat_fit, p_value_fit = stats.ttest_rel(ga_fitnesses, pso_fitnesses)
    print(f"1️⃣  FITNESS COMPARISON")
    print(f"   T-statistic: {t_stat_fit:.4f}")
    print(f"   P-value: {p_value_fit:.6f}")
    print(f"   Result: {'❌ TIDAK signifikan (p > 0.05)' if p_value_fit > 0.05 else '✅ SIGNIFIKAN (p < 0.05)'}")
    print(f"   Winner: {'PSO' if np.mean(pso_fitnesses) > np.mean(ga_fitnesses) else 'GA'} "
          f"(Mean: {'PSO=' + f'{np.mean(pso_fitnesses):.6f}' if np.mean(pso_fitnesses) > np.mean(ga_fitnesses) else 'GA=' + f'{np.mean(ga_fitnesses):.6f}'})")
    
    # Test 2: Cost
    t_stat_cost, p_value_cost = stats.ttest_rel(ga_costs, pso_costs)
    print(f"\n2️⃣  COST COMPARISON")
    print(f"   T-statistic: {t_stat_cost:.4f}")
    print(f"   P-value: {p_value_cost:.6f}")
    print(f"   Result: {'❌ TIDAK signifikan (p > 0.05)' if p_value_cost > 0.05 else '✅ SIGNIFIKAN (p < 0.05)'}")
    print(f"   Winner: {'PSO' if np.mean(pso_costs) < np.mean(ga_costs) else 'GA'} "
          f"(Mean: {'PSO=Rp' + f'{np.mean(pso_costs):,.0f}' if np.mean(pso_costs) < np.mean(ga_costs) else 'GA=Rp' + f'{np.mean(ga_costs):,.0f}'})")
    
    # Test 3: Time
    t_stat_time, p_value_time = stats.ttest_rel(ga_times, pso_times)
    print(f"\n3️⃣  COMPUTATIONAL TIME COMPARISON")
    print(f"   T-statistic: {t_stat_time:.4f}")
    print(f"   P-value: {p_value_time:.6f}")
    print(f"   Result: {'❌ TIDAK signifikan (p > 0.05)' if p_value_time > 0.05 else '✅ SIGNIFIKAN (p < 0.05)'}")
    print(f"   Winner: {'PSO' if np.mean(pso_times) < np.mean(ga_times) else 'GA'} "
          f"(Mean: {'PSO=' + f'{np.mean(pso_times):.2f}s' if np.mean(pso_times) < np.mean(ga_times) else 'GA=' + f'{np.mean(ga_times):.2f}s'})")
    
    # === CONFIDENCE INTERVALS (95%) ===
    print(f"\n{'='*70}")
    print(f"{'📊 95% CONFIDENCE INTERVALS':^70}")
    print(f"{'='*70}\n")
    
    # CI formula: mean ± t_critical * (std / sqrt(n))
    confidence_level = 0.95
    degrees_freedom = n_runs - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    # GA CI
    ga_fit_ci = t_critical * (np.std(ga_fitnesses, ddof=1) / np.sqrt(n_runs))
    ga_cost_ci = t_critical * (np.std(ga_costs, ddof=1) / np.sqrt(n_runs))
    ga_time_ci = t_critical * (np.std(ga_times, ddof=1) / np.sqrt(n_runs))
    
    print(f"🧬 GENETIC ALGORITHM")
    print(f"   Fitness : {np.mean(ga_fitnesses):.6f} ± {ga_fit_ci:.6f}")
    print(f"   Cost    : Rp {np.mean(ga_costs):,.0f} ± {ga_cost_ci:,.0f}")
    print(f"   Time    : {np.mean(ga_times):.2f}s ± {ga_time_ci:.2f}s")
    
    # PSO CI
    pso_fit_ci = t_critical * (np.std(pso_fitnesses, ddof=1) / np.sqrt(n_runs))
    pso_cost_ci = t_critical * (np.std(pso_costs, ddof=1) / np.sqrt(n_runs))
    pso_time_ci = t_critical * (np.std(pso_times, ddof=1) / np.sqrt(n_runs))
    
    print(f"\n🌊 PARTICLE SWARM OPTIMIZATION")
    print(f"   Fitness : {np.mean(pso_fitnesses):.6f} ± {pso_fit_ci:.6f}")
    print(f"   Cost    : Rp {np.mean(pso_costs):,.0f} ± {pso_cost_ci:,.0f}")
    print(f"   Time    : {np.mean(pso_times):.2f}s ± {pso_time_ci:.2f}s")
    
    # === BOXPLOT VISUALIZATION ===
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
    print(f"\n✅ Statistical plots saved to: {plot_path}")
    
    plt.show()
    
    print(f"\n{'='*70}\n")
    
    # === RETURN RESULTS ===
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

# ============================================================================
# 9. MAIN PROGRAM - MENU UTAMA
# ============================================================================

def main():
    """
    Main program dengan menu interaktif
    """
    print(f"\n{'='*70}")
    print(f"{'🍽️  NUTRITION OPTIMIZATION SYSTEM':^70}")
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
        print(f"5. Exit")
        print(f"{'─'*70}")
        
        choice = input("\nPilih menu (1-5): ").strip()
        
        if choice == '1':
            # === SINGLE DAY COMPARISON ===
            print(f"\n{'='*70}")
            print(f"{'🔥 SINGLE DAY OPTIMIZATION':^70}")
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
            print("\n🧬 GA - BEST MENU")
            print_menu_detail(ga_solution)
            
            print("\n🌊 PSO - BEST MENU")
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
            # === WEEKLY MENU GENERATION ===
            print("\n📅 Generate menu untuk 7 hari")
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
                    print(f"✅ GA results saved to: {ga_file}")
                
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
                    print(f"✅ PSO results saved to: {pso_file}")
        
        elif choice == '3':
            # === STATISTICAL ANALYSIS ===
            print("\n📊 Statistical Analysis")
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
                print(f"✅ Statistical results saved to: {json_file}")
        
        elif choice == '4':
            # === QUICK DEMO ===
            print(f"\n{'='*70}")
            print(f"{'⚡ QUICK DEMO - BEST SOLUTION':^70}")
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
            
            print(f"\n🏆 WINNER: {winner}")
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
            # === EXIT ===
            print(f"\n{'='*70}")
            print(f"{'👋 Terima kasih telah menggunakan program ini!':^70}")
            print(f"{'='*70}\n")
            break
        
        else:
            print("\n❌ Pilihan tidak valid! Silakan pilih 1-5.")

# ============================================================================
# 10. ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def export_menu_to_pdf(solution: np.ndarray, filename: str = "menu.txt"):
    """
    Export menu detail ke text file (bisa dikonversi ke PDF)
    
    Args:
        solution: Array 50 elemen
        filename: Nama file output
    """
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
    
    print(f"✅ Menu exported to: {filename}")

def compare_multiple_runs(n_runs: int = 10, algorithm: str = 'both'):
    """
    Compare multiple runs untuk melihat stabilitas algoritma
    
    Args:
        n_runs: Jumlah runs
        algorithm: 'ga', 'pso', atau 'both'
    
    Returns: Comparison data
    """
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
    
    print(f"\n✅ Completed {n_runs} runs!\n")
    
    # Calculate coefficient of variation (CV) = std/mean
    if algorithm in ['ga', 'both']:
        ga_cv = np.std(results['GA']) / np.mean(results['GA']) * 100
        print(f"🧬 GA Stability:")
        print(f"   Mean: {np.mean(results['GA']):.6f}")
        print(f"   Std:  {np.std(results['GA']):.6f}")
        print(f"   CV:   {ga_cv:.2f}%")
    
    if algorithm in ['pso', 'both']:
        pso_cv = np.std(results['PSO']) / np.mean(results['PSO']) * 100
        print(f"\n🌊 PSO Stability:")
        print(f"   Mean: {np.mean(results['PSO']):.6f}")
        print(f"   Std:  {np.std(results['PSO']):.6f}")
        print(f"   CV:   {pso_cv:.2f}%")
    
    print(f"\nInterpretation:")
    print(f"  • CV < 10%  : Very stable ✅")
    print(f"  • CV 10-20% : Stable")
    print(f"  • CV > 20%  : Less stable ⚠️")
    
    return results

# ============================================================================
# 11. RUN PROGRAM
# ============================================================================

if __name__ == "__main__":
    """
    Program entry point
    
    Usage:
    1. Run interaktif: python nutrition_optimization.py
    2. Run langsung demo: Uncomment baris di bawah
    """
    
    # === OPTION 1: INTERACTIVE MODE (DEFAULT) ===
    main()
    
    # === OPTION 2: DIRECT EXECUTION (Uncomment untuk auto-run) ===
    # print("Running Quick Demo...")
    # ga = GeneticAlgorithm(pop_size=30, generations=50, pc=0.8, pm=0.2)
    # ga_solution, ga_fitness, ga_history = ga.evolve(verbose=True)
    # print_menu_detail(ga_solution)
    
    # pso = ParticleSwarmOptimization(n_particles=30, iterations=50, w=0.7, c1=1.5, c2=1.5)
    # pso_solution, pso_fitness, pso_history = pso.optimize(verbose=True)
    # print_menu_detail(pso_solution)
    
    # plot_convergence_comparison(ga_history, pso_history)

# ============================================================================
# 12. DOKUMENTASI TAMBAHAN
# ============================================================================

"""
DOKUMENTASI LENGKAP PROGRAM

1. STRUKTUR PROGRAM
   - Section 1: Dataset (50 makanan Indonesia)
   - Section 2: Nutritional targets
   - Section 3: Fitness function & constraints
   - Section 4: Genetic Algorithm implementation
   - Section 5: PSO implementation
   - Section 6: Helper functions (reporting & visualization)
   - Section 7: Weekly menu generation
   - Section 8: Statistical analysis (30 runs + t-test)
   - Section 9: Main program (interactive menu)
   - Section 10: Additional utilities
   - Section 11: Program entry point
   - Section 12: Documentation

2. CARA MENJALANKAN
   a. Interactive Mode:
      python nutrition_optimization.py
      
   b. Jupyter Notebook:
      - Copy semua code ke satu cell
      - Run cell
      - Pilih menu
   
   c. Command Line Arguments (Advanced):
      # Edit di section 11 untuk custom behavior

3. OUTPUT FILES
   - weekly_menu_GA_TIMESTAMP.csv: Menu 7 hari (GA)
   - weekly_menu_PSO_TIMESTAMP.csv: Menu 7 hari (PSO)
   - statistical_analysis_TIMESTAMP.png: Boxplot statistik
   - statistical_results_TIMESTAMP.json: Raw statistical data
   - menu.txt: Exported menu (text format)

4. PARAMETER TUNING GUIDELINES
   
   GA Parameters:
   - pop_size: 20-50 (default=30)
     • Lebih besar = eksplorasi lebih baik, tapi lebih lambat
   - generations: 30-100 (default=50)
     • Lebih banyak = solusi lebih optimal
   - pc (crossover rate): 0.7-0.9 (default=0.8)
     • Tinggi = lebih banyak eksplorasi
   - pm (mutation rate): 0.1-0.3 (default=0.2)
     • Tinggi = lebih diverse, hindari premature convergence
   
   PSO Parameters:
   - n_particles: 20-50 (default=30)
     • Sama seperti GA pop_size
   - iterations: 30-100 (default=50)
     • Sama seperti GA generations
   - w (inertia): 0.4-0.9 (default=0.7)
     • Tinggi = eksplorasi, rendah = eksploitasi
   - c1 (cognitive): 1.0-2.0 (default=1.5)
     • Pengaruh personal best
   - c2 (social): 1.0-2.0 (default=1.5)
     • Pengaruh global best

5. INTERPRETASI HASIL
   
   a. Fitness Value:
      • Higher is better
      • Typical range: 0.025 - 0.035
      • Formula: 1000 / (cost + penalty + 1)
   
   b. Penalty:
      • Zero penalty = semua constraints terpenuhi ✅
      • High penalty = banyak constraint violation ❌
   
   c. T-test P-value:
      • p < 0.05: Perbedaan signifikan secara statistik
      • p > 0.05: Tidak ada perbedaan signifikan
   
   d. Convergence:
      • Fast convergence: Baik untuk efisiensi
      • Slow convergence: Mungkin lebih thorough exploration

6. TROUBLESHOOTING
   
   Problem: Penalty selalu tinggi
   Solution: 
   - Relax constraints di TARGETS
   - Increase population/particles
   - Increase generations/iterations
   
   Problem: Cost selalu > budget
   Solution:
   - Increase MAX_BUDGET
   - Adjust penalty weights (increase w₄)
   
   Problem: Algoritma lambat
   Solution:
   - Reduce pop_size/n_particles
   - Reduce generations/iterations
   - Run in parallel (advanced)

7. CUSTOMIZATION TIPS
   
   a. Menambah makanan baru:
      • Edit FOOD_DATABASE (section 1)
      • Update NUM_FOODS otomatis
   
   b. Mengubah target nutrisi:
      • Edit TARGETS (section 2)
      • Sesuaikan dengan kebutuhan spesifik
   
   c. Mengubah penalty weights:
      • Edit calculate_penalty() (section 3)
      • Contoh: protein penting → increase weight dari 2.0 ke 5.0
   
   d. Menambah constraint baru:
      • Tambahkan logika di calculate_penalty()
      • Contoh: constraint untuk vitamin, mineral, dll

8. PERBANDINGAN GA vs PSO (Expected Results)
   
   Genetic Algorithm:
   ✅ Pros:
      • Robust terhadap local optima
      • Good for discrete/combinatorial problems
      • Natural diversity maintenance
   ❌ Cons:
      • Potentially slower convergence
      • More parameters to tune
   
   Particle Swarm Optimization:
   ✅ Pros:
      • Fast convergence
      • Fewer parameters
      • Simple implementation
   ❌ Cons:
      • Risk of premature convergence
      • Less effective for discrete problems
   
   Expected Outcomes:
   • Quality: Similar (PSO mungkin sedikit lebih baik)
   • Speed: PSO biasanya lebih cepat
   • Stability: GA biasanya lebih stabil

9. VALIDASI HASIL
   
   Checklist sebelum menggunakan menu:
   ☐ Kalori dalam range 1800-2200 kkal
   ☐ Protein ≥ 50g
   ☐ Karbohidrat 250-350g
   ☐ Budget ≤ Rp 50,000
   ☐ Setiap kategori memenuhi minimum portion
   ☐ Tidak ada makanan dengan porsi aneh (>500g)

10. REFERENCES & CITATIONS
    
    Algoritma:
    • Genetic Algorithm: Holland, J.H. (1975)
    • PSO: Kennedy & Eberhart (1995)
    
    Data:
    • TKPI (Tabel Komposisi Pangan Indonesia) 2017
    • Permenkes RI No. 28 Tahun 2019
    
    Optimization:
    • Constraint handling: Penalty function method
    • Multi-objective: Weighted sum approach

11. ADVANCED FEATURES (Future Work)
    
    Bisa ditambahkan:
    • Multi-objective optimization (cost + diversity + taste)
    • Constraint handling: Feasibility rules
    • Hybrid GA-PSO
    • Parallel processing
    • Deep learning untuk prediksi preferensi
    • Real-time price update dari API
    • User preference learning
    • Allergen constraints
    • Meal timing optimization

12. CONTACT & SUPPORT
    
    Developer: Mochamad Faisal Akbar (L0122094)
    Course: Kecerdasan Komputasional
    University: Universitas Sebelas Maret
    
    Untuk pertanyaan atau bug report:
    • Email: [your_email]
    • GitHub: [your_github]

END OF DOCUMENTATION
"""

# ============================================================================
# 13. EXAMPLE USAGE SCENARIOS
# ============================================================================

"""
CONTOH PENGGUNAAN PROGRAM

Scenario 1: Mahasiswa Kost (Budget Terbatas)
──────────────────────────────────────────────
Target:
- Budget: Rp 30,000/hari
- Protein: Minimal (50g)
- Prioritas: Murah tapi sehat

Modifikasi:
1. Set MAX_BUDGET = 30000
2. Increase penalty weight untuk budget (0.01 → 0.05)
3. Run program

Expected Result:
- Banyak tempe, tahu, telur (protein murah)
- Nasi putih, mie (karbohidrat murah)
- Sayur lokal (bayam, kangkung)

──────────────────────────────────────────────

Scenario 2: Atlet/Bodybuilder (High Protein)
──────────────────────────────────────────────
Target:
- Protein: 100g+ per hari
- Kalori: 2500-3000 kkal
- Budget: Lebih fleksibel

Modifikasi:
1. Set TARGETS['protein']['min'] = 100
2. Set TARGETS['kalori']['max'] = 3000
3. Increase protein penalty weight (2.0 → 5.0)
4. Set MAX_BUDGET = 80000

Expected Result:
- Banyak ayam, ikan, telur
- Susu, yogurt
- Karbohidrat kompleks (oatmeal, roti gandum)

──────────────────────────────────────────────

Scenario 3: Diet (Weight Loss)
──────────────────────────────────────────────
Target:
- Kalori: 1500-1800 kkal
- Protein: Tinggi (60g+)
- Karbo: Rendah (150-200g)

Modifikasi:
1. Set TARGETS['kalori'] = {'min': 1500, 'max': 1800, 'ideal': 1650}
2. Set TARGETS['karbo']['max'] = 200
3. Set TARGETS['protein']['min'] = 60

Expected Result:
- Banyak protein (ayam, ikan, telur)
- Sayuran (rendah kalori)
- Buah (portion controlled)
- Karbohidrat minimal

──────────────────────────────────────────────

Scenario 4: Vegetarian
──────────────────────────────────────────────
Target:
- Tanpa daging & ikan
- Protein dari nabati
- Nutrisi seimbang

Modifikasi:
1. Hapus ayam, ikan, daging dari FOOD_DATABASE
2. Tambah lebih banyak kacang-kacangan
3. Increase tempe, tahu portion weights

Expected Result:
- Tempe, tahu sebagai protein utama
- Kacang merah, kacang hijau
- Banyak sayur & buah
- Susu/susu kedelai

──────────────────────────────────────────────
"""

# ============================================================================
# 14. TESTING & VALIDATION FUNCTIONS
# ============================================================================

def validate_solution(solution: np.ndarray) -> Dict:
    """
    Validasi apakah solusi memenuhi semua constraints
    
    Args:
        solution: Array 50 elemen
    
    Returns: Dictionary validation results
    """
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

def test_algorithms():
    """
    Unit testing untuk GA dan PSO
    Memastikan algoritma berjalan dengan benar
    """
    print(f"\n{'='*70}")
    print(f"{'🧪 TESTING ALGORITHMS':^70}")
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
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 2: GA evolution
    print("Test 2: GA Evolution...", end=' ')
    try:
        ga = GeneticAlgorithm(pop_size=10, generations=5)
        solution, fitness, history = ga.evolve(verbose=False)
        assert len(solution) == NUM_FOODS
        assert fitness > 0
        assert len(history['best_history']) == 5
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 3: PSO initialization
    print("Test 3: PSO Initialization...", end=' ')
    try:
        pso = ParticleSwarmOptimization(n_particles=10, iterations=5)
        positions = np.random.uniform(0, 300, (10, NUM_FOODS))
        assert positions.shape == (10, NUM_FOODS)
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 4: PSO optimization
    print("Test 4: PSO Optimization...", end=' ')
    try:
        pso = ParticleSwarmOptimization(n_particles=10, iterations=5)
        solution, fitness, history = pso.optimize(verbose=False)
        assert len(solution) == NUM_FOODS
        assert fitness > 0
        assert len(history['best_history']) == 5
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Fitness function
    print("Test 5: Fitness Function...", end=' ')
    try:
        test_solution = np.random.uniform(0, 300, NUM_FOODS)
        fitness = fitness_function(test_solution)
        assert fitness > 0
        assert np.isfinite(fitness)
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Constraint validation
    print("Test 6: Constraint Validation...", end=' ')
    try:
        test_solution = np.random.uniform(100, 200, NUM_FOODS)
        validation = validate_solution(test_solution)
        assert 'valid' in validation
        assert 'penalty' in validation
        assert 'constraints' in validation
        print("✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*70}\n")
    
    return tests_passed, tests_failed

# ============================================================================
# 15. PERFORMANCE PROFILING
# ============================================================================

def profile_performance():
    """
    Profile performance algoritma untuk optimasi
    """
    import time
    
    print(f"\n{'='*70}")
    print(f"{'⏱️  PERFORMANCE PROFILING':^70}")
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
        print(f"⚡ PSO is {speedup:.2f}x faster than GA")
    else:
        print(f"⚡ GA is {1/speedup:.2f}x faster than PSO")
    
    return results

# ============================================================================
# 16. FINAL NOTES & TIPS
# ============================================================================

"""
FINAL NOTES FOR IMPLEMENTATION

✅ CHECKLIST SEBELUM DEMO/PRESENTASI:
   1. [ ] Test semua fungsi dengan test_algorithms()
   2. [ ] Run statistical analysis minimal 30x
   3. [ ] Generate menu 7 hari untuk showcase
   4. [ ] Save semua output (CSV, PNG, JSON)
   5. [ ] Prepare backup data jika demo gagal
   6. [ ] Print beberapa menu contoh untuk ditunjukkan
   7. [ ] Siapkan penjelasan matematika (lihat document 1)

📊 TIPS PRESENTASI:
   1. Mulai dengan penjelasan problem (4 Sehat 5 Sempurna)
   2. Jelaskan decision variables & constraints
   3. Tunjukkan convergence plot (GA vs PSO)
   4. Highlight hasil statistical test (t-test)
   5. Showcase menu 7 hari yang realistic
   6. Diskusikan trade-offs (cost vs nutrition)
   7. Conclude dengan winner (GA atau PSO)

🎯 EXPECTED QUESTIONS & ANSWERS:
   Q: Kenapa PSO lebih baik/GA lebih baik?
   A: [Tergantung hasil] Jelaskan dari segi convergence, stability, time
   
   Q: Apakah menu ini realistis?
   A: Ya, semua makanan ada di pasar Indonesia, harga real
   
   Q: Kenapa tidak multi-objective?
   A: Simplifikasi untuk fokus ke perbandingan algoritma
   
   Q: Bagaimana handle infeasible solutions?
   A: Penalty function method, dijelaskan di calculate_penalty()
   
   Q: Apakah bisa ditambah constraint lain?
   A: Ya, tinggal tambah di calculate_penalty() (contoh: vitamin)

💡 IMPROVEMENT IDEAS (Jika ada waktu):
   1. Add visualisasi pie chart komposisi nutrisi
   2. Add meal timing (sarapan, makan siang, makan malam)
   3. Add variety score (hindari makanan sama setiap hari)
   4. Add taste preference model
   5. Add seasonal availability constraint
   6. Integrate real-time price API

🐛 KNOWN LIMITATIONS:
   1. Harga statis (tidak real-time)
   2. Tidak consider cooking time/effort
   3. Tidak consider food combinations (taste)
   4. Single objective (bisa diperbaiki dengan NSGA-II)
   5. Tidak ada constraint untuk variety antar hari

📚 RECOMMENDED READING:
   1. Genetic Algorithms: David Goldberg
   2. PSO: Kennedy & Eberhart original paper
   3. Constraint handling: Deb's feasibility rules
   4. Multi-objective: NSGA-II paper

GOOD LUCK WITH YOUR PRESENTATION! 🚀
"""

# ============================================================================
# END OF PROGRAM
# ============================================================================

print("\n" + "="*70)
print(" ✅ Program loaded successfully!")
print(" 📘 Read documentation above for usage guide")
print(" 🚀 Run main() to start interactive menu")
print("="*70 + "\n")