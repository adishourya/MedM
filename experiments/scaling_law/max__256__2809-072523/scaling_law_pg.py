# this is to get the number of epochs i would need to get desired loss
# but we will go context window 512 when we get dsri..
# and the time changes too... dsri has more cores , hbm ...[but this is just to practice how i will do it in future]

# power law equation
# loss_t = a (t^(-b)) + c

# if c is small
# log(loss) = log(a) + log(t^-b) = log(a) -b(log(t))
# then we can use linear regrssion to get the log loss

# sklearn doesnt have power law fit
epochs_step = 4_000
predict_till = 5 * epochs_step

from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import matplotx

# Load your dataset
run = pd.read_csv("./run__max256__2809-072523.csv")
print(run.columns)  # Check the column names

# Extract step (as time, t) and value (loss)
steps = run['Step'].values  # Independent variable (time or epochs)
loss_values = run['Value'].values  # Dependent variable (loss)

# Power law equation
def power_law(t, a, b, c):
    return a * (t ** (-b)) + c

# Fit the power law to the data
# Initial guesses for a, b, and c
initial_guesses = [1.0, 0.5, 0.1]
popt, pcov = curve_fit(power_law, steps, loss_values, p0=initial_guesses)

# Extract the fitted parameters
a, b, c = popt
print(f"Fitted parameters: a={a}, b={b}, c={c}")

# Generate a smooth curve for plotting the fitted model
t_fit = np.linspace(min(steps), max(steps)+ predict_till, 500)
loss_fit = power_law(t_fit, a, b, c)


# Function to estimate the number of epochs needed to reach the desired loss
def epochs_to_reach_loss(desired_loss, a, b, c):
    # Solve for t: desired_loss = a * (t ** (-b)) + c
    if desired_loss >= c:
        print("Desired loss is too high, the loss cannot converge to that value.")
        return None
    t_estimate = ((desired_loss - c) / a) ** (-1 / b)
    return t_estimate

# Estimate how many steps/epochs are needed to reach the desired loss
desired_loss = 3.5  # Example target loss
estimated_steps = epochs_to_reach_loss(desired_loss, a, b, c)
if estimated_steps:
    print(f"Estimated steps/epochs to reach loss {desired_loss}: {estimated_steps:.2f}")

# Plot the original data and the fitted curve
plt.style.use(matplotx.styles.pacoty)
plt.figure(figsize=(8, 6))
plt.scatter(steps, loss_values, label='Actual', color='blue', s=10)
plt.plot(t_fit, loss_fit, label=f'Fitted Power Law\n$Loss(t) = {a:.4f}t^{{-{b:.4f}}} + {c:.4f}$', color='red')
plt.hlines(y= desired_loss,xmin=0 , xmax = predict_till, linestyle='dashed', color='green', label='Desired Loss')
plt.xlabel('Steps (Epochs or Batches)')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Power Law Fit to Loss Data... steps needed to reach {desired_loss=} is {estimated_steps=}')
plt.savefig("bad_news_lorapg1.png")
plt.show()
