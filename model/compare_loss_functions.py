"""
Author: Thai Ha Bui
Description: This script is designed to compare the training losses of two different loss functions for a binary classification task. It loads the training losses from two different directories and performs a t-test to determine if there is a significant difference between the two loss functions.
"""

import os
import torch
import matplotlib.pyplot as plt
from brainspy.utils.pytorch import TorchUtils
from scipy.stats import ttest_ind

def load_losses(results_dir):
    training_data = torch.load(os.path.join(results_dir, 'reproducibility', 'training_data.pickle'), map_location=TorchUtils.get_device())
    return training_data['train_losses'] 

corrsig = "output/boolean_gd/[0, 1, 1, 0]_2024_06_26_005834 (corrsig)"
bce = "output/boolean_gd/[0, 1, 1, 0]_2024_06_26_002539 (new loss)"

losses_corrsig = load_losses(corrsig)
losses_bce = load_losses(bce)

# Perform t-test
t_stat, p_value = ttest_ind(losses_corrsig, losses_bce)

print(f"T-statistic: {t_stat}, P-value: {p_value}")

plt.figure(figsize=(10, 5))
plt.plot(losses_corrsig, label='Correlation Sigmoid')
plt.plot(losses_bce, label='Piecewise Linear')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Training Losses Comparison')
plt.legend()
plt.show()