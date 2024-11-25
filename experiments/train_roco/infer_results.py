import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotx
plt.style.use(matplotx.styles.pacoty)

results = pd.read_json("./test_results.json")
print(results.columns)

# rouge scores
# ROUGE-1: Measures the overlap of unigrams (individual words) between the generated and reference texts. It helps assess general relevance. [0.4-0.6 is treated as good]
# ROUGE-2: Measures the overlap of bigrams (pairs of words), capturing more contextual accuracy than ROUGE-1 and indicating how well the model maintains meaningful word pairs. [0.2-0.4]
# ROUGE-L: Based on the longest common subsequence (LCS), it identifies the longest sequence of words in the correct order, which is particularly useful for capturing fluency and syntactic structure. [0.3-0.5]

plt.hist(results.rouge1, label="rouge1")
plt.hist(results.rouge2, label="rouge2")
plt.hist(results.rougeL, label="rougeL")
plt.legend()
plt.title("Rouge Metrics")
plt.show()

