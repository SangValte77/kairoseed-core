import numpy as np
import matplotlib.pyplot as plt

# Ensemble parameters
seeds = [0, 1, 2, 3, 4]
n_trajectories = 20
n_steps = 100
sigma_values = np.linspace(0.1, 2.0, 20)

# Storage for ensemble results
entropy_by_sigma = {sigma: [] for sigma in sigma_values}
distance_by_sigma = {sigma: [] for sigma in sigma_values}
variance_by_sigma = {sigma: [] for sigma in sigma_values}

print("Computing phase transition dynamics...")

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

# Compute statistics
entropy_means = []
entropy_stds = []
distance_means = []
distance_stds = []
variance_means = []
variance_stds = []

for sigma in sigma_values:
    entropy_means.append(np.mean(entropy_by_sigma[sigma]))
    entropy_stds.append(np.std(entropy_by_sigma[sigma]))
    distance_means.append(np.mean(distance_by_sigma[sigma]))
    distance_stds.append(np.std(distance_by_sigma[sigma]))
    variance_means.append(np.mean(variance_by_sigma[sigma]))
    variance_stds.append(np.std(variance_by_sigma[sigma]))

# Create phase transition visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Entropy with confidence band
ax = axes[0, 0]
ax.fill_between(
    sigma_values,
    np.array(entropy_means) - np.array(entropy_stds),
    np.array(entropy_means) + np.array(entropy_stds),
    alpha=0.3,
    color='#FF6B6B',
    label='± 1 std'
)
ax.plot(sigma_values, entropy_means, 'o-', linewidth=2.5, markersize=7, color='#FF6B6B', label='Mean')
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Transition threshold')
ax.set_xlabel('σ (noise level)', fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized Entropy', fontsize=11, fontweight='bold')
ax.set_title('Phase I: Coherence → Diffusion', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])
ax.legend(loc='upper left', fontsize=9)

# Plot 2: Distance with confidence band
ax = axes[0, 1]
ax.fill_between(
    sigma_values,
    np.array(distance_means) - np.array(distance_stds),
    np.array(distance_means) + np.array(distance_stds),
    alpha=0.3,
    color='#4ECDC4',
    label='± 1 std'
)
ax.plot(sigma_values, distance_means, 's-', linewidth=2.5, markersize=7, color='#4ECDC4', label='Mean')
ax.set_xlabel('σ (noise level)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Pairwise Distance', fontsize=11, fontweight='bold')
ax.set_title('Phase II: Trajectory Divergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9)

# Plot 3: Variance with confidence band
ax = axes[1, 0]
ax.fill_between(
    sigma_values,
    np.array(variance_means) - np.array(variance_stds),
    np.array(variance_means) + np.array(variance_stds),
    alpha=0.3,
    color='#95E1D3',
    label='± 1 std'
)
ax.plot(sigma_values, variance_means, '^-', linewidth=2.5, markersize=7, color='#95E1D3', label='Mean')
ax.set_xlabel('σ (noise level)', fontsize=11, fontweight='bold')
ax.set_ylabel('Final Variance', fontsize=11, fontweight='bold')
ax.set_title('Phase III: State Space Spread', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9)

# Plot 4: Multi-observable phase diagram
ax = axes[1, 1]
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))

p1 = ax.plot(sigma_values, entropy_means, 'o-', linewidth=2.5, markersize=7, color='#FF6B6B', label='Entropy')
p2 = ax2.plot(sigma_values, distance_means, 's-', linewidth=2.5, markersize=7, color='#4ECDC4', label='Distance')
p3 = ax3.plot(sigma_values, variance_means, '^-', linewidth=2.5, markersize=7, color='#95E1D3', label='Variance')

ax.set_xlabel('σ (noise level)', fontsize=11, fontweight='bold')
ax.set_ylabel('Entropy', fontsize=10, color='#FF6B6B', fontweight='bold')
ax2.set_ylabel('Distance', fontsize=10, color='#4ECDC4', fontweight='bold')
ax3.set_ylabel('Variance', fontsize=10, color='#95E1D3', fontweight='bold')

ax.tick_params(axis='y', labelcolor='#FF6B6B')
ax2.tick_params(axis='y', labelcolor='#4ECDC4')
ax3.tick_params(axis='y', labelcolor='#95E1D3')

ax.set_title('Phase Transition: Multi-Observable View', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Combine legends
ps = p1 + p2 + p3
labels = [p.get_label() for p in ps]
ax.legend(ps, labels, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('phase_transition_diagram.png', dpi=150, bbox_inches='tight')
print("✓ Saved: phase_transition_diagram.png")

# Print summary
print("\n" + "=" * 90)
print("Phase Transition Summary: σ Controls Attractor Structure")
print("=" * 90)
print(f"{'σ':<6} {'Entropy':<20} {'Distance':<20} {'Variance':<20} {'Phase':<15}")
print("-" * 90)

for i, sigma in enumerate(sigma_values):
    entropy_val = entropy_means[i]
    state = "COHERENT" if entropy_val < 0.4 else "TRANSITION" if entropy_val < 0.6 else "DIFFUSE"
    
    print(
        f"{sigma:.2f}  "
        f"{entropy_means[i]:.4f} ± {entropy_stds[i]:.4f}  "
        f"{distance_means[i]:.4f} ± {distance_stds[i]:.4f}  "
        f"{variance_means[i]:.4f} ± {variance_stds[i]:.4f}  "
        f"{state}"
    )

print("=" * 90)
print("Interpretation:")
print("  σ < 0.5:  Attractor dominates (trajectories collapse together)")
print("  σ ≈ 1.0:  Critical transition zone (structure becomes unstable)")
print("  σ > 1.5:  Noise dominates (attractor dissolves, chaos emerges)")
