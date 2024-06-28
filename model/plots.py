import matplotlib.pyplot as plt
import numpy as np

# Data for V2 and V4 voltages
time = np.linspace(0, 0.4, 400)
V2 = np.where((time > 0.2) & (time <= 0.4), 0.6, -1.2)
V4 = np.where((time > 0.1) & (time <= 0.2) | (time > 0.3) & (time <= 0.4), 0.6, -1.2)

# Data for prediction and target
time_sim = np.linspace(0, 40, 400)
prediction = np.where(time_sim > 30, 2.0, -1.5)
target = np.where(time_sim > 30, 0.5, 0.0)

# Plotting the applied voltages
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax[0].plot(time, V2, color='goldenrod')
ax[0].set_ylabel('V2 (V)')
ax[0].set_ylim([-1.5, 0.8])
ax[0].text(0.1, -0.3, '0', ha='center', va='center')
ax[0].text(0.3, -0.3, '1', ha='center', va='center')
ax[0].text(0.35, -0.3, '1', ha='center', va='center')

ax[1].plot(time, V4, color='blue')
ax[1].set_ylabel('V4 (V)')
ax[1].set_ylim([-1.5, 0.8])
ax[1].set_xlabel('Time (s)')
ax[1].text(0.1, -0.3, '0', ha='center', va='center')
ax[1].text(0.2, -0.3, '1', ha='center', va='center')
ax[1].text(0.35, -0.3, '1', ha='center', va='center')

plt.tight_layout()
plt.show()