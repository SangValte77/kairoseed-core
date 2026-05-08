import numpy as np
from scipy.stats import entropy as scipy_entropy

np.random.seed(42)

# Dynamics parameters
W = np.random.randn(10, 10) * 0.5  # Weight matrix for bounded contraction
n_trajectories = 20
n_steps = 100
sigma_values = np.linspace(0.1, 2.0, 20)

print("Entropy Decay Observable")
print("=" * 70)
print(f"Trajectories: {n_trajectories} | Steps: {n_steps}")
print("=" * 70)

for sigma in sigma_values:
    # Initialize random trajectories
    trajectories = np.random.randn(n_trajectories, 10)
    
    # Evolve trajectories under dynamics
    for step in range(n_steps):
        noise = np.random.randn(n_trajectories, 10) * sigma
        trajectories = np.tanh(trajectories @ W.T + noise)
    
    # Compute state distribution entropy
    # Discretize state space into bins for entropy calculation
    n_bins = 10
    bins = np.linspace(-1, 1, n_bins + 1)
    
    # Flatten trajectories and compute histogram
    flat_trajectories = trajectories.flatten()
    hist, _ = np.histogram(flat_trajectories, bins=bins)
    
    # Normalize to get probability distribution
    prob_dist = hist / hist.sum()
    
    # Remove zero probabilities for entropy calculation
    prob_dist = prob_dist[prob_dist > 0]
    
    # Calculate Shannon entropy
    shannon_entropy = -np.sum(prob_dist * np.log2(prob_dist))
    
    # Normalize entropy to [0, 1] range
    max_entropy = np.log2(len(prob_dist))
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
    
    # Compute final pairwise distances
    distances = []
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            dist = np.linalg.norm(trajectories[i] - trajectories[j])
            distances.append(dist)
    
    mean_distance = np.mean(distances)
    final_variance = np.var(trajectories)
    
    # State classification
    state = "COHERENT" if normalized_entropy < 0.5 else "DIFFUSE"
    
    print(
        f"sigma={sigma:.2f} | "
        f"entropy={normalized_entropy:.4f} | "
        f"distance={mean_distance:.4f} | "
        f"variance={final_variance:.4f} | "
        f"state={state}"
    )

print("=" * 70)
print("Observable: Entropy increases → Trajectories diffuse as σ increases")
print("Low entropy (COHERENT) ≡ Trajectories clustered")
print("High entropy (DIFFUSE) ≡ Trajectories scattered")
