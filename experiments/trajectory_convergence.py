import numpy as np

np.random.seed(42)

# Dynamics parameters
W = np.random.randn(10, 10) * 0.5  # Weight matrix for bounded contraction
n_trajectories = 20
n_steps = 100
sigma_values = np.linspace(0.1, 2.0, 20)

print("Trajectory Convergence Experiment")
print("=" * 60)
print(f"Trajectories: {n_trajectories} | Steps: {n_steps}")
print("=" * 60)

for sigma in sigma_values:
    # Initialize random trajectories
    trajectories = np.random.randn(n_trajectories, 10)
    
    # Evolve trajectories under dynamics
    for step in range(n_steps):
        noise = np.random.randn(n_trajectories, 10) * sigma
        trajectories = np.tanh(trajectories @ W.T + noise)
    
    # Compute final pairwise distances
    distances = []
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            dist = np.linalg.norm(trajectories[i] - trajectories[j])
            distances.append(dist)
    
    mean_distance = np.mean(distances)
    final_variance = np.var(trajectories)
    
    # State classification
    state = "COHERENT" if mean_distance < 1.0 else "DIFFUSE"
    
    print(
        f"sigma={sigma:.2f} | "
        f"mean_pairwise_dist={mean_distance:.4f} | "
        f"variance={final_variance:.4f} | "
        f"state={state}"
    )

print("=" * 60)
print("Observable: Trajectories converge → diverge as σ increases")
