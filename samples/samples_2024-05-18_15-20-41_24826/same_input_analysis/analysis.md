# Simulation Analysis Report

## Random Outputs Statistical Measures:
- **Mean**: $ \mu = 0.0021034613227256 $
- **Standard Deviation**: \( \sigma = 0.030111091800599996 \)
- **Variance**: \( \sigma^2 = 0.0009066778494241603 \)
- **Range**: \( R = 0.113163 \)
- **Median**: \( \tilde{x} = -0.000935501 \)
- **Interquartile Range (IQR)**: \( \text{IQR} = 0.055761599999999995 \)

## Same Input Outputs Statistical Measures:
- **Mean**: \( \mu = -0.0158382626 \)
- **Standard Deviation**: \( \sigma = 0.001095113203257631 \)
- **Variance**: \( \sigma^2 = 1.1992729279491892 \times 10^{-6} \)
- **Range**: \( R = 0.006708699999999998 \)
- **Median**: \( \tilde{x} = -0.0158493 \)
- **Interquartile Range (IQR)**: \( \text{IQR} = 0.0014099000000000004 \)

## Analysis:

### 1. Mean Comparison:
- The mean of the random outputs (\( \mu = 0.0021 \)) is close to zero, indicating that the outputs are centered around zero.
- The mean of the same input outputs (\( \mu = -0.0158 \)) is slightly negative, but this is expected as it represents the average output for a specific set of inputs.

### 2. Standard Deviation and Variance:
- The standard deviation and variance of the random outputs (\( \sigma = 0.0301 \) and \( \sigma^2 = 0.00091 \), respectively) are significantly higher than those of the same input outputs (\( \sigma = 0.0011 \) and \( \sigma^2 = 1.199 \times 10^{-6} \), respectively). This indicates that the random outputs have a much wider spread, as expected.
- The low standard deviation and variance for the same input outputs suggest that the outputs are very consistent when the same inputs are applied.

### 3. Range:
- The range of the random outputs (\( R = 0.113163 \)) is much larger than the range of the same input outputs (\( R = 0.0067087 \)). This further supports the observation that the same input outputs are tightly clustered around their mean value.

### 4. Median and IQR:
- The median values for both random and same input outputs are close to their respective means, indicating symmetric distributions.
- The IQR for the random outputs (\( \text{IQR} = 0.05576 \)) is much larger than the IQR for the same input outputs (\( \text{IQR} = 0.00141 \)), indicating that the middle 50% of the same input outputs are very close to each other.

### 5. Confidence Intervals:
- For the random outputs, a 95% confidence interval for the mean can be calculated as:
  $$
  \mu \pm 1.96 \frac{\sigma}{\sqrt{n}} = 0.0021 \pm 1.96 \frac{0.0301}{\sqrt{10000}} = 0.0021 \pm 0.00059
  $$
- For the same input outputs, a 95% confidence interval for the mean can be calculated as:
  $$
  \mu \pm 1.96 \frac{\sigma}{\sqrt{n}} = -0.0158 \pm 1.96 \frac{0.0011}{\sqrt{1000}} = -0.0158 \pm 0.000068
  $$

### 6. Hypothesis Testing:
- To test if the mean of the same input outputs is significantly different from zero, we can perform a one-sample t-test:
  $$
  t = \frac{\mu - 0}{\sigma / \sqrt{n}} = \frac{-0.0158}{0.0011 / \sqrt{1000}} = -453.64
  $$
  Given the large t-value, we reject the null hypothesis that the mean is zero.

## Conclusion:
The statistical measures indicate that the simulation is consistent and robust. When the same input voltages are applied over many samples, the output current values do not vary much, as evidenced by the low standard deviation, variance, and range. This suggests that the simulation produces reliable and repeatable results for the same inputs, which is a desirable property for any simulation.

In summary, the simulation appears to be consistent and robust, as the outputs for the same inputs show minimal variation, while the outputs for random inputs exhibit a wider range of values. This consistency is crucial for ensuring the reliability of the simulation in practical applications.