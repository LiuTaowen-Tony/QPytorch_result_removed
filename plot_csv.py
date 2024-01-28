import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('results/loss_scale_batchnorm_clip_experiment/result.csv')

# Step 2: Preprocess the Data
# Ensure that 'activation_round', 'batchnorm', 'loss_scale', and 'test_loss' are in the correct format

print(df.keys())

df["test_loss"] = df["test_loss"].astype(float)
# Step 3: Plot the Graph
# Create a unique identifier for each combination of 'activation_round' and 'batchnorm'
df['group'] = df['activation_round'].astype(str) + ' x ' + df['batchnorm'] + ' x ' + df['clip'].astype(str)
plt.ylim(1.5, 2.3)


# Plot each group
for name, group in df.groupby('group'):
    plt.scatter(np.log(group['loss_scale']), group['test_loss'], label=name)

# Step 4: Customize the Plot
plt.xlabel('Loss Scale')
plt.ylabel('Test Loss')
plt.title('Test Loss vs Loss Scale for Different Activation Rounds and Batchnorms')
plt.legend()

# Show the plot
plt.savefig('plot.png')
