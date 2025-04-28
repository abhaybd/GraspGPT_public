import csv
import os

res_folder = "/results" if os.path.isdir("/results") else "gcngrasp/results"
results_files = os.listdir(res_folder)
results_files = [f for f in results_files if f.endswith("results.csv")]

splitmode_accuracies = {}
results = {}
for results_file in results_files:
    # open the results csv which is run_name and accuracy
    with open(os.path.join(res_folder, results_file), "r") as f:
        reader = csv.reader(f)
        
        for row in reader:
            run_name = row[0]
            accuracy = float(row[1].strip("%"))
            mean_ap = float(row[2].strip("%"))
            results[run_name] = (accuracy, mean_ap)

# run name is splitmode_splitid. We want to average the accuracies for each splitmode

for run_name, (accuracy, mean_ap) in results.items():
    splitmode = run_name.split("_")[0]
    if splitmode not in splitmode_accuracies:
        splitmode_accuracies[splitmode] = []
    splitmode_accuracies[splitmode].append((accuracy, mean_ap))

print(splitmode_accuracies)
# average the accuracies for each splitmode
for splitmode, acc_ap in splitmode_accuracies.items():
    accuracies = [acc for acc, _ in acc_ap]
    mean_aps = [ap for _, ap in acc_ap]
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_mean_ap = sum(mean_aps) / len(mean_aps)
    print("Avg Acc")
    print(f"{splitmode}: {avg_accuracy:.4f}")
    
    print("Avg mAP")
    print(f"{splitmode}: {avg_mean_ap:.4f}")

'''
# write the final results to a new csv file in final directory
# this should be file_name, splitmode, accuracy
final_results_file = os.path.join("gcngrasp", "results", "final", "final_results.csv")

with open(final_results_file, "a") as f:
    writer = csv.writer(f)
    # write the header if the file is empty
    if os.stat(final_results_file).st_size == 0:
        writer.writerow(["file_name", "splitmode", "accuracy"])
    for splitmode, accuracies in splitmode_accuracies.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        writer.writerow([results_file, splitmode, avg_accuracy])
'''