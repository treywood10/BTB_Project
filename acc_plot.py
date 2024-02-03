import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import bourbon csv #
historical = pd.read_csv('historical_model_comparisons.csv')

# Accuracy #
my_acc = np.mean(historical['My_Prediction'] == historical['Bourbon_today'])
their_acc = np.mean(historical['Their_Prediction'] == historical['Bourbon_today'])

# Define a function for bootstrap resampling
def bootstrap_accuracy(data, model_column, target_column, num_iterations=5000):
    accuracies = []
    for _ in range(num_iterations):
        # Resample with replacement
        bootstrap_sample = data.sample(n=len(data), replace=True)

        # Calculate accuracy for the resampled data
        accuracy = np.mean(bootstrap_sample[model_column] == bootstrap_sample[target_column])
        accuracies.append(accuracy)

    # Calculate confidence intervals
    lower_bound = np.percentile(accuracies, 2.5)
    upper_bound = np.percentile(accuracies, 97.5)

    return lower_bound, upper_bound


# Bootstrap accuracy and confidence intervals for 'My_Prediction' and 'Their_Prediction'
my_lower, my_upper = bootstrap_accuracy(historical, 'My_Prediction', 'Bourbon_today')
their_lower, their_upper = bootstrap_accuracy(historical, 'Their_Prediction', 'Bourbon_today')

# Dataframe #
df = pd.DataFrame({
    'Model': ['My_Prediction', 'Their_Prediction'],
    'Accuracy': [my_acc, their_acc],
    'Lower_CI_Bootstrap': [my_lower, their_lower],
    'Upper_CI_Bootstrap': [my_upper, their_upper]
})

# Assuming you already have the 'df' DataFrame with accuracy and bootstrap CIs
model_names = df['Model']
accuracies = df['Accuracy'] * 100  # Convert accuracy to percentage
lower_cis = df['Lower_CI_Bootstrap'] * 100
upper_cis = df['Upper_CI_Bootstrap'] * 100

# Set up plot #
fig, ax = plt.subplots(figsize=(8, 6))

# Bar plot for accuracies #
ax.bar(model_names, accuracies, color=['blue', 'orange'], alpha=0.7, label='Accuracy')

# Error bars for bootstrap confidence intervals
for i, model in enumerate(model_names):
    ax.errorbar(model, accuracies[i], yerr=[[accuracies[i] - lower_cis[i]], [upper_cis[i] - accuracies[i]]],
                fmt='none', color='black', capsize=5, capthick=2)

# Adding labels and title #
ax.set_ylabel('Accuracy (%)')  # Update y-axis label
ax.set_title('Model Performance', fontsize = 12)

# Set y-axis ticks to represent percentages #
ax.set_yticks(np.arange(0, 101, 10))

# Customize x-axis tick labels #
ax.set_xticks(model_names)
ax.set_xticklabels(['My Model', 'Their Model'])

# Turn off the legend
ax.legend().set_visible(False)

# Show the plot
plt.savefig('Acc_plot.png')
plt.show()