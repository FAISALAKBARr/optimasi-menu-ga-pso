from scipy import stats

# Run 30 kali untuk dapat data statistik
ga_results = []
pso_results = []

for run in range(30):
    # Set random seed berbeda tiap run
    np.random.seed(run)
    
    # Run GA
    ga = GeneticAlgorithm()
    ga_solution, _ = ga.run()
    ga_nutrition = calculate_nutrition(ga_solution)
    ga_results.append(ga_nutrition['cost'])
    
    # Run PSO
    pso = ParticleSwarmOptimization()
    pso_solution, _ = pso.run()
    pso_nutrition = calculate_nutrition(pso_solution)
    pso_results.append(pso_nutrition['cost'])

# T-test
t_stat, p_value = stats.ttest_ind(ga_results, pso_results)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Perbedaan signifikan secara statistik (p < 0.05)")
    if np.mean(ga_results) < np.mean(pso_results):
        print("GA significantly better")
    else:
        print("PSO significantly better")
else:
    print("❌ Tidak ada perbedaan signifikan (p ≥ 0.05)")