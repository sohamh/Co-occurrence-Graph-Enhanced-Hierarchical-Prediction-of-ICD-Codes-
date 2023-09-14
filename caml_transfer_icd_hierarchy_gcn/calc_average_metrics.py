
# Function to calculate the average and variance of test metrics
def calculate_average_and_variance_metrics(metrics_files):
    total_metrics = len(metrics_files)
    average_metrics = {}
    variance_metrics = {}
    f1_micro_values = []
    f1_macro_values = []
    prec_8_values = []

    # Initialize average_metrics and variance_metrics dictionaries with zeros
    with open(metrics_files[0]) as f:
        data = json.load(f)
        for metric, value in data.items():
            if metric.endswith("_te"):
                average_metrics[metric] = 0
                variance_metrics[metric] = 0

    # Accumulate metric values from all files
    for metrics_file in metrics_files:
        if "._metrics" in metrics_file:
            continue
        with open(metrics_file) as f:
            data = json.load(f)
            f1_micro_values.append(data["f1_micro_te"])
            f1_macro_values.append(data["f1_macro_te"])
            prec_8_values.append(data["prec_at_8_te"])
            for metric, value in data.items():
                if metric.endswith("_te"):
                    average_metrics[metric] += value[0]
                    variance_metrics[metric] += value[0] ** 2


    # Calculate average by dividing accumulated values by total metrics
    for metric in average_metrics:
        average_metrics[metric] /= total_metrics

    # Calculate variance
    for metric in variance_metrics:
        variance_metrics[metric] = (variance_metrics[metric] / total_metrics) - (average_metrics[metric] ** 2)
        variance_metrics[metric] = math.sqrt(variance_metrics[metric])

    return average_metrics, variance_metrics, f1_micro_values, f1_macro_values, prec_8_values

# Provide the path to the directory containing the JSON files
directory_path = "/home/mahdi/codes/caml_transfer_icd_hierarchy_gcn/avg_compute_caml_gc"

# Get all JSON files in the directory
metrics_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".json")]

# Calculate the average and variance of the metrics
average_metrics_1, variance_metrics_1, f1_micro_1, f1_macro_1, prec_8_1 = calculate_average_and_variance_metrics(metrics_files)


directory_path = "/home/mahdi/codes/caml_transfer_icd_hierarchy_gcn/avg_compute_caml"

# Get all JSON files in the directory
metrics_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".json")]

# Calculate the average and variance of the metrics
average_metrics_2, variance_metrics_2, f1_micro_2, f1_macro_2, prec_8_2 = calculate_average_and_variance_metrics(metrics_files)

# Create a list of lists for the table headers and values
table_headers = ["Metric", "Average", "Variance"]
table_values = []

src=["f1_micro_te", "f1_macro_te", "prec_at_8_te"]
# Populate the table_values with the metric names, average values, and variance values
for metric in average_metrics_1:
    if metric in src:
        table_values.append([metric, average_metrics_1[metric], variance_metrics_1[metric]])

# Generate the table using the tabulate library
table = tabulate(table_values, headers=table_headers, tablefmt="grid")

# Print the table
print(table)

# Example data for Model A and Model B performance (replace with your actual data)
model_A = f1_macro_1  # Performance of Model A
model_B = f1_macro_2  # Performance of Model B

# Perform the paired t-test
t_statistic, p_value_ttest = stats.ttest_rel(model_A, model_B)

# Perform the paired Wilcoxon signed-rank test
z_statistic, p_value_wilcoxon = stats.wilcoxon(model_A, model_B)

# Compare p-values to significance level
alpha = 0.05

if p_value_ttest < alpha:
    print("The difference in performance between the two models is statistically significant (paired t-test).")
else:
    print("The difference in performance between the two models is not statistically significant (paired t-test).")

if p_value_wilcoxon < alpha:
    print("The difference in performance between the two models is statistically significant (paired Wilcoxon test).")
else:
    print("The difference in performance between the two models is not statistically significant (paired Wilcoxon test).")
