import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Global formatting
# -------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18

# -------------------------
# Activation functions
# -------------------------
def identity(x):
    return x

def logistic(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# -------------------------
# Domain
# -------------------------
x = np.linspace(-5, 5, 1000)

# -------------------------
# Create 2x2 subplot
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

functions = [
    ("Identity", identity),
    ("Logistic (Sigmoid)", logistic),
    ("Tanh", tanh),
    ("ReLU", relu)
]

# -------------------------
# Plot
# -------------------------
for ax, (title, func) in zip(axes, functions):

    y = func(x)

    ax.plot(x, y, color='red', linewidth=2.5)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    ax.grid(True)

    ax.axhline(0, linewidth=1, color='black')
    ax.axvline(0, linewidth=1, color='black')

plt.tight_layout()

# -------------------------
# Export to PDF (vector)
# -------------------------
plt.savefig(
    "activation_functions.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()