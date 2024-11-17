import numpy as np
import matplotlib.pyplot as plt


# Parameters for the Noam learning rate scheduler
model_size = 128
factor = 0.2
warmup_steps = 2000
total_steps = 20000  # Example total number of steps to plot

# Noam learning rate schedule calculation
steps = np.arange(1, total_steps + 1)
lr = factor * (model_size ** (-0.5) * np.minimum(steps ** (-0.5), steps * (warmup_steps ** (-1.5))))

# Plotting the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(steps, lr)
plt.title('Learning Rate Schedule (Noam)')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()
