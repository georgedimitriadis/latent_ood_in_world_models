import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data / 255.0  # Normalize pixel values

# Perform NMF
n_components = 49  # Number of components to extract
nmf = NMF(n_components=n_components, init='random', random_state=0, verbose=True)
W = nmf.fit_transform(X)
X_hat = nmf.inverse_transform(W)
H = nmf.components_

print("mse", mean_squared_error(X, X_hat))

# Visualize the components
fig, axes = plt.subplots(7, 7, figsize=(15, 15))
fig.suptitle("NMF Components", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < n_components:
        ax.imshow(H[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    else:
        ax.remove()

plt.tight_layout()
plt.savefig("./nmf.pdf")

# Visualize reconstructions of some digits
n_samples = 5
fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
fig.suptitle("Original vs Reconstructed Digits", fontsize=16)

for i in range(n_samples):
    original = X[i].reshape(28, 28)
    reconstructed = np.dot(W[i], H).reshape(28, 28)

    axes[0, i].imshow(original, cmap='gray')
    axes[0, i].set_title(f"Original {i + 1}")
    axes[0, i].axis('off')

    axes[1, i].imshow(reconstructed, cmap='gray')
    axes[1, i].set_title(f"Reconstructed {i + 1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()