import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Dynamics parameters
W = np.random.randn(10, 10) * 0.5  # Weight matrix for bounded contraction
n_trajectories = 20
n_steps = 100
sigma_values = np.linspace(0.1, 2.0, 20)

# Storage for results
entropy_results = []
distance_results = []
variance_results = []

print("Computing entropy curve...")

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
    
    # Compute pairwise distances
    distances = []
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            dist = np.linalg.norm(trajectories[i] - trajectories[j])
            distances.append(dist)
    
    mean_distance = np.mean(distances)
    final_variance = np.var(trajectories)
    
    entropy_results.append(normalized_entropy)
    distance_results.append(mean_distance)
    variance_results.append(final_variance)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Entropy vs Sigma
axes[0].plot(sigma_values, entropy_results, 'o-', linewidth=2, markersize=6, color='#FF6B6B')
axes[0].set_xlabel('σ (noise level)', fontsize=11)
axes[0].set_ylabel('Normalized Entropy', fontsize=11)
axes[0].set_title('Entropy: COHERENT → DIFFUSE', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Plot 2: Mean Pairwise Distance vs Sigma
axes[1].plot(sigma_values, distance_results, 's-', linewidth=2, markersize=6, color='#4ECDC4')
axes[1].set_xlabel('σ (noise level)', fontsize=11)
axes[1].set_ylabel('Mean Pairwise Distance', fontsize=11)
axes[1].set_title('Trajectory Divergence', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Variance vs Sigma
axes[2].plot(sigma_values, variance_results, '^-', linewidth=2, markersize=6, color='#95E1D3')
axes[2].set_xlabel('σ (noise level)', fontsize=11)
axes[2].set_ylabel('Final Variance', fontsize=11)
axes[2].set_title('State Space Spread', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attractor_transition_curves.png', dpi=150, bbox_inches='tight')
print("✓ Saved: attractor_transition_curves.png")

# Print data table
print("\n" + "=" * 80)
print("Observable Summary: σ Transition Dynamics")
print("=" * 80)
print(f"{'σ':<6} {'Entropy':<12} {'Distance':<12} {'Variance':<12} {'State':<10}")
print("-" * 80)
for i, sigma in enumerate(sigma_values):
    state = "COHERENT" if entropy_results[i] < 0.5 else "DIFFUSE"
    print(
        f"{sigma:.2f}  {entropy_results[i]:.4f}       {distance_results[i]:.4f}       "
        f"{variance_results[i]:.4f}       {state}"
    )
print("=" * 80)
