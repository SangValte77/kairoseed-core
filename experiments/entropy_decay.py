import numpy as np

# Ensemble parameters
seeds = [0, 1, 2, 3, 4]
n_trajectories = 20
n_steps = 100
sigma_values = np.linspace(0.1, 2.0, 20)

# Storage for ensemble results
entropy_by_sigma = {sigma: [] for sigma in sigma_values}
distance_by_sigma = {sigma: [] for sigma in sigma_values}
variance_by_sigma = {sigma: [] for sigma in sigma_values}

print("Entropy Decay Observable: Ensemble Robustness Test")
print("=" * 80)
print(f"Seeds: {seeds} | Trajectories: {n_trajectories} | Steps: {n_steps}")
print("=" * 80)

for seed in seeds:
    np.random.seed(seed)
    
    # Dynamics parameters (reinitialized per seed)
    W = np.random.randn(10, 10) * 0.5  # Weight matrix for bounded contraction
    
    for sigma in sigma_values:
        # Initialize random trajectories
        trajectories = np.random.randn(n_trajectories, 10)
        
        # Evolve trajectories under dynamics
        for step in range(n_steps):
            noise = np.random.randn(n_trajectories, 10) * sigma
            trajectories = np.tanh(trajectories @ W.T + noise)
        
        # Compute state distribution entropy
        n_bins = 10
        bins = np.linspace(-1, 1, n_bins + 1)
        
        flat_trajectories = trajectories.flatten()
        hist, _ = np.histogram(flat_trajectories, bins=bins)
        
        prob_dist = hist / hist.sum()
        prob_dist = prob_dist[prob_dist > 0]
        
        shannon_entropy = -np.sum(prob_dist * np.log2(prob_dist))
        max_entropy = np.log2(n_bins)
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        # Compute final pairwise distances
        distances = []
        for i in range(n_trajectories):
            for j in range(i + 1, n_trajectories):
                dist = np.linalg.norm(trajectories[i] - trajectories[j])
                distances.append(dist)
        
        mean_distance = np.mean(distances)
        final_variance = np.var(trajectories)
        
        # Store results
        entropy_by_sigma[sigma].append(normalized_entropy)
        distance_by_sigma[sigma].append(mean_distance)
        variance_by_sigma[sigma].append(final_variance)

# Compute statistics and display
print(f"\n{'σ':<6} {'Entropy':<20} {'Distance':<20} {'Variance':<20} {'State':<10}")
print("-" * 80)

for sigma in sigma_values:
    entropy_mean = np.mean(entropy_by_sigma[sigma])
    entropy_std = np.std(entropy_by_sigma[sigma])
    
    distance_mean = np.mean(distance_by_sigma[sigma])
    distance_std = np.std(distance_by_sigma[sigma])
    
    variance_mean = np.mean(variance_by_sigma[sigma])
    variance_std = np.std(variance_by_sigma[sigma])
    
    state = "COHERENT" if entropy_mean < 0.5 else "DIFFUSE"
    
    print(
        f"{sigma:.2f}  "
        f"{entropy_mean:.4f} ± {entropy_std:.4f}  "
        f"{distance_mean:.4f} ± {distance_std:.4f}  "
        f"{variance_mean:.4f} ± {variance_std:.4f}  "
        f"{state}"
    )

print("=" * 80)
print("Observable: Ensemble statistics show structural persistence across seeds")
print("Low entropy (COHERENT) ≡ Trajectories clustered")
print("High entropy (DIFFUSE) ≡ Trajectories scattered")
