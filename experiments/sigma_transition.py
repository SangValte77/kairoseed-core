import numpy as np

np.random.seed(42)

sigma_values = np.linspace(0.1, 2.0, 20)

print("Sigma Transition Experiment")
print("-" * 40)

for sigma in sigma_values:
    x = np.random.randn(1000) * sigma
    entropy_proxy = np.var(x)

    state = "COHERENT" if entropy_proxy < 1.0 else "DIFFUSE"

    print(
        f"sigma={sigma:.2f} | "
        f"variance={entropy_proxy:.4f} | "
        f"state={state}"
    )
