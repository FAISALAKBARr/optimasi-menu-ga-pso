# ============================================================================
# NUTRITION OPTIMIZATION USING GA & PSO
# Project: Optimasi Menu Makanan Seimbang
# Author: Mochamad Faisal Akbar
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import os
from datetime import datetime

# ============================================================================
# 1. DATASET - 50 Makanan Indonesia (10 per kategori)
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
ALL_FOODS = []
CATEGORY_START = {}
idx = 0
for category, foods in FOOD_DATABASE.items():
    CATEGORY_START[category] = idx
    for food in foods:
        food['category'] = category
        ALL_FOODS.append(food)
    idx += len(foods)

NUM_FOODS = len(ALL_FOODS)
print(f"Total makanan dalam database: {NUM_FOODS}")

# ============================================================================
# 2. NUTRITIONAL TARGETS
# ============================================================================

TARGETS = {
    'kalori': {'min': 1800, 'max': 2200, 'ideal': 2000},
    'protein': {'min': 50, 'max': 80, 'ideal': 60},
    'karbo': {'min': 250, 'max': 350, 'ideal': 300}
}

CATEGORY_MIN_PORTIONS = {
    'buah': 150,        # gram
    'karbohidrat': 300,
    'protein': 150,
    'sayur': 200,
    'minuman': 200
}

MAX_BUDGET = 50000  # Rp per hari

# ============================================================================
# 3. FITNESS FUNCTION
# ============================================================================

def calculate_nutrition(portions: np.ndarray) -> Dict:
    """Calculate total nutrition from portions"""
    total_kalori = 0
    total_protein = 0
    total_karbo = 0
    total_cost = 0
    
    for i, portion in enumerate(portions):
        food = ALL_FOODS[i]
        # Nutrisi per 100g
        total_kalori += food['kalori'] * (portion / 100)
        total_protein += food['protein'] * (portion / 100)
        total_karbo += food['karbo'] * (portion / 100)
        total_cost += food['harga'] * (portion / 1000)  # harga per kg
    
    return {
        'kalori': total_kalori,
        'protein': total_protein,
        'karbo': total_karbo,
        'cost': total_cost
    }

def calculate_penalty(portions: np.ndarray, nutrition: Dict) -> float:
    """Calculate constraint violations"""
    penalty = 0
    
    # Nutrition constraints
    if nutrition['kalori'] < TARGETS['kalori']['min']:
        penalty += (TARGETS['kalori']['min'] - nutrition['kalori']) ** 2 * 0.1
    if nutrition['kalori'] > TARGETS['kalori']['max']:
        penalty += (nutrition['kalori'] - TARGETS['kalori']['max']) ** 2 * 0.1
    
    if nutrition['protein'] < TARGETS['protein']['min']:
        penalty += (TARGETS['protein']['min'] - nutrition['protein']) ** 2 * 2
    
    if nutrition['karbo'] < TARGETS['karbo']['min']:
        penalty += (TARGETS['karbo']['min'] - nutrition['karbo']) ** 2 * 0.05
    if nutrition['karbo'] > TARGETS['karbo']['max']:
        penalty += (nutrition['karbo'] - TARGETS['karbo']['max']) ** 2 * 0.05
    
    # Budget constraint
    if nutrition['cost'] > MAX_BUDGET:
        penalty += (nutrition['cost'] - MAX_BUDGET) ** 2 * 0.01
    
    # Category minimum portions
    for category, min_portion in CATEGORY_MIN_PORTIONS.items():
        start_idx = CATEGORY_START[category]
        end_idx = start_idx + len(FOOD_DATABASE[category])
        category_total = np.sum(portions[start_idx:end_idx])
        
        if category_total < min_portion:
            penalty += (min_portion - category_total) ** 2 * 0.5
    
    return penalty

def fitness_function(portions: np.ndarray) -> float:
    """Main fitness function (higher is better)"""
    nutrition = calculate_nutrition(portions)
    penalty = calculate_penalty(portions, nutrition)
    
    # Objective: minimize cost
    cost = nutrition['cost']
    
    # Fitness = 1 / (cost + penalty + 1)
    fitness = 1000 / (cost + penalty + 1)
    
    return fitness

# ============================================================================
# 4. GENETIC ALGORITHM (GA)
# ============================================================================

class GeneticAlgorithm:
    def __init__(self, pop_size=30, generations=50, pc=0.8, pm=0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc  # crossover rate
        self.pm = pm  # mutation rate
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self):
        """Create initial population"""
        population = []
        for _ in range(self.pop_size):
            # Random portions 0-300g
            individual = np.random.uniform(0, 300, NUM_FOODS)
            population.append(individual)
        return population
    
    def tournament_selection(self, population, fitnesses, k=3):
        """Tournament selection"""
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.random() < self.pc:
            point = np.random.randint(1, NUM_FOODS)
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
            return offspring1, offspring2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Gaussian mutation"""
        for i in range(NUM_FOODS):
            if np.random.random() < self.pm:
                individual[i] += np.random.normal(0, 30)
                individual[i] = np.clip(individual[i], 0, 500)
        return individual
    
    def run(self):
        """Main GA loop"""
        print("\n" + "="*60)
        print("GENETIC ALGORITHM")
        print("="*60)
        
        start_time = time.time()
        
        # Initialize
        population = self.initialize_population()
        
        for gen in range(self.generations):
            # Evaluate
            fitnesses = [fitness_function(ind) for ind in population]
            
            # Track progress
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]
            best_nutrition = calculate_nutrition(best_individual)
            
            if gen % 10 == 0:
                print(f"Gen {gen:3d} | Best Fitness: {best_fitness:.2f} | "
                      f"Cost: Rp {best_nutrition['cost']:.0f} | "
                      f"Cal: {best_nutrition['kalori']:.0f}")
            
            # Create next generation
            new_population = []
            
            # Elitism (keep top 10%)
            elite_count = self.pop_size // 10
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            population = new_population[:self.pop_size]
        
        # Final evaluation
        fitnesses = [fitness_function(ind) for ind in population]
        best_idx = np.argmax(fitnesses)
        best_solution = population[best_idx]
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ GA Completed in {elapsed_time:.2f} seconds")
        
        return best_solution, self.best_fitness_history

# ============================================================================
# 5. PARTICLE SWARM OPTIMIZATION (PSO)
# ============================================================================

class ParticleSwarmOptimization:
    def __init__(self, swarm_size=20, iterations=50, w=0.9, c1=2.0, c2=2.0):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w_start = w
        self.w_end = 0.4
        self.c1 = c1
        self.c2 = c2
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_swarm(self):
        """Initialize particles"""
        positions = []
        velocities = []
        
        for _ in range(self.swarm_size):
            pos = np.random.uniform(0, 300, NUM_FOODS)
            vel = np.zeros(NUM_FOODS)
            positions.append(pos)
            velocities.append(vel)
        
        return positions, velocities
    
    def run(self):
        """Main PSO loop"""
        print("\n" + "="*60)
        print("PARTICLE SWARM OPTIMIZATION")
        print("="*60)
        
        start_time = time.time()
        
        # Initialize
        positions, velocities = self.initialize_swarm()
        pbest_positions = [p.copy() for p in positions]
        pbest_fitnesses = [fitness_function(p) for p in positions]
        
        gbest_idx = np.argmax(pbest_fitnesses)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitnesses[gbest_idx]
        
        for iter in range(self.iterations):
            # Update inertia weight (linear decay)
            w = self.w_start - (self.w_start - self.w_end) * (iter / self.iterations)
            
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.random(NUM_FOODS), np.random.random(NUM_FOODS)
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Velocity clamping
                velocities[i] = np.clip(velocities[i], -50, 50)
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 500)
                
                # Evaluate
                fitness = fitness_function(positions[i])
                
                # Update pbest
                if fitness > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                
                # Update gbest
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()
            
            # Track progress
            avg_fitness = np.mean(pbest_fitnesses)
            self.best_fitness_history.append(gbest_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            if iter % 10 == 0:
                best_nutrition = calculate_nutrition(gbest_position)
                print(f"Iter {iter:3d} | Best Fitness: {gbest_fitness:.2f} | "
                      f"Cost: Rp {best_nutrition['cost']:.0f} | "
                      f"Cal: {best_nutrition['kalori']:.0f}")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ PSO Completed in {elapsed_time:.2f} seconds")
        
        return gbest_position, self.best_fitness_history

# ============================================================================
# 6. HASIL & VISUALISASI
# ============================================================================

def print_solution(solution: np.ndarray, algorithm_name: str):
    """Print detailed solution"""
    print("\n" + "="*60)
    print(f"{algorithm_name} - BEST SOLUTION")
    print("="*60)
    
    nutrition = calculate_nutrition(solution)
    
    print(f"\n💰 Total Biaya: Rp {nutrition['cost']:.0f}/hari")
    print(f"🍽️  Total Kalori: {nutrition['kalori']:.0f} kcal")
    print(f"🥩 Protein: {nutrition['protein']:.1f}g")
    print(f"🍚 Karbohidrat: {nutrition['karbo']:.1f}g")
    
    print("\n📋 DAFTAR MAKANAN (porsi > 10g):")
    print("-" * 60)
    
    for category in FOOD_DATABASE.keys():
        category_foods = []
        start_idx = CATEGORY_START[category]
        
        for i, food in enumerate(FOOD_DATABASE[category]):
            portion = solution[start_idx + i]
            if portion > 10:  # Show only significant portions
                category_foods.append((food['nama'], portion))
        
        if category_foods:
            print(f"\n{category.upper()} ({NUTRITION_GUIDE[category]['emoji']}):")
            for nama, portion in category_foods:
                print(f"  - {nama:20s}: {portion:6.1f}g")

def plot_comparison(ga_history, pso_history):
    """Plot convergence comparison"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ga_history, label='GA', linewidth=2)
    plt.plot(pso_history, label='PSO', linewidth=2)
    plt.xlabel('Generation/Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Normalize to percentage
    ga_normalized = (np.array(ga_history) / ga_history[-1]) * 100
    pso_normalized = (np.array(pso_history) / pso_history[-1]) * 100
    
    plt.plot(ga_normalized, label='GA', linewidth=2)
    plt.plot(pso_normalized, label='PSO', linewidth=2)
    plt.xlabel('Generation/Iteration')
    plt.ylabel('% of Final Best Fitness')
    plt.title('Normalized Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
    
    plt.tight_layout()

    # Ensure results directory exists (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, 'convergence_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Grafik disimpan: {out_path}")


def save_solution_report(ga_solution, ga_history, ga_nutrition,
                         pso_solution, pso_history, pso_nutrition):
    """Save a human-readable solution report into results/ folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(results_dir, 'solution_report.txt')
    report_path_ts = os.path.join(results_dir, f'solution_report_{timestamp}.txt')

    def format_solution(solution):
        lines = []
        for category in FOOD_DATABASE.keys():
            start_idx = CATEGORY_START[category]
            items = []
            for i, food in enumerate(FOOD_DATABASE[category]):
                portion = float(solution[start_idx + i])
                if portion > 10:
                    items.append(f"{food['nama']}: {portion:.1f} g")
            if items:
                lines.append(f"{category.upper()}:")
                for it in items:
                    lines.append(f"  - {it}")
        return "\n".join(lines)

    content_lines = []
    content_lines.append("OPTIMASI MENU MAKANAN - SOLUTION REPORT")
    content_lines.append(f"Generated: {datetime.now().isoformat()}")
    content_lines.append("\n-- GENETIC ALGORITHM (GA) --")
    content_lines.append(f"Final Fitness: {ga_history[-1]:.4f}")
    content_lines.append(f"Total Cost (Rp): {ga_nutrition['cost']:.2f}")
    content_lines.append(f"Kalori (kcal): {ga_nutrition['kalori']:.2f}")
    content_lines.append(f"Protein (g): {ga_nutrition['protein']:.2f}")
    content_lines.append(f"Karbo (g): {ga_nutrition['karbo']:.2f}")
    content_lines.append("\nDAFTAR MAKANAN (porsi > 10g):")
    content_lines.append(format_solution(ga_solution))

    content_lines.append("\n-- PARTICLE SWARM OPTIMIZATION (PSO) --")
    content_lines.append(f"Final Fitness: {pso_history[-1]:.4f}")
    content_lines.append(f"Total Cost (Rp): {pso_nutrition['cost']:.2f}")
    content_lines.append(f"Kalori (kcal): {pso_nutrition['kalori']:.2f}")
    content_lines.append(f"Protein (g): {pso_nutrition['protein']:.2f}")
    content_lines.append(f"Karbo (g): {pso_nutrition['karbo']:.2f}")
    content_lines.append("\nDAFTAR MAKANAN (porsi > 10g):")
    content_lines.append(format_solution(pso_solution))

    # Write both a latest copy and a timestamped archive
    full_text = "\n".join(content_lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    with open(report_path_ts, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"\n✅ Laporan solusi disimpan: {report_path}")
    print(f"✅ Arsip laporan: {report_path_ts}")

# Add nutrition guide reference
NUTRITION_GUIDE = {
    'buah': {'emoji': '🍎'},
    'karbohidrat': {'emoji': '🍚'},
    'protein': {'emoji': '🍖'},
    'sayur': {'emoji': '🥬'},
    'minuman': {'emoji': '🥤'}
}

# ============================================================================
# 7. MAIN PROGRAM
# ============================================================================

def main():
    print("="*60)
    print("OPTIMASI MENU MAKANAN - GA vs PSO")
    print("Mochamad Faisal Akbar (L0122094) - Kecerdasan Komputasional")
    print("="*60)
    
    # Run GA
    ga = GeneticAlgorithm(pop_size=30, generations=50)
    ga_solution, ga_history = ga.run()
    print_solution(ga_solution, "GENETIC ALGORITHM")
    
    # Run PSO
    pso = ParticleSwarmOptimization(swarm_size=20, iterations=50)
    pso_solution, pso_history = pso.run()
    print_solution(pso_solution, "PARTICLE SWARM OPTIMIZATION")
    
    # Comparison
    print("\n" + "="*60)
    print("PERBANDINGAN GA vs PSO")
    print("="*60)
    
    ga_nutrition = calculate_nutrition(ga_solution)
    pso_nutrition = calculate_nutrition(pso_solution)
    
    comparison_data = {
        'Metric': ['Total Cost (Rp)', 'Kalori (kcal)', 'Protein (g)', 'Karbo (g)', 
                   'Final Fitness', 'Convergence Speed'],
        'GA': [f"{ga_nutrition['cost']:.0f}", 
               f"{ga_nutrition['kalori']:.0f}",
               f"{ga_nutrition['protein']:.1f}",
               f"{ga_nutrition['karbo']:.1f}",
               f"{ga_history[-1]:.2f}",
               f"Gen {len([x for x in ga_history if x < ga_history[-1]*0.95])}"],
        'PSO': [f"{pso_nutrition['cost']:.0f}",
                f"{pso_nutrition['kalori']:.0f}",
                f"{pso_nutrition['protein']:.1f}",
                f"{pso_nutrition['karbo']:.1f}",
                f"{pso_history[-1]:.2f}",
                f"Iter {len([x for x in pso_history if x < pso_history[-1]*0.95])}"]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\n", df.to_string(index=False))
    
    # Plot
    plot_comparison(ga_history, pso_history)
    # Save textual solution report into results/
    try:
        save_solution_report(ga_solution, ga_history, ga_nutrition,
                             pso_solution, pso_history, pso_nutrition)
    except Exception as e:
        print(f"Gagal menyimpan laporan solusi: {e}")
    
    print("\n✅ Program selesai!")
    print("📊 Hasil tersimpan dalam grafik 'convergence_comparison.png'")

if __name__ == "__main__":
    main()