import csv
import os
from collections import defaultdict
import pdb

res_folder = "/net/nfs2.prior/arijitr/research/semantic_grasping/GraspGPT_public/gcngrasp/results"
results_files = os.listdir("/net/nfs2.prior/arijitr/research/semantic_grasping/GraspGPT_public/gcngrasp/results")
results_files = [f for f in results_files if f.endswith("_levels.csv")]

splitmode_accuracies = defaultdict(lambda: defaultdict(list))
results = defaultdict(list)
for results_file in results_files:
    # open the results csv which is run_name and accuracy
    with open(os.path.join(res_folder, results_file), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            run_name = row[0]
            level = row[1]
            accuracy = float(row[2].strip("%"))
            results[run_name].append((accuracy, level))

print(results)

# run name is splitmode_splitid. We want to average the accuracies for each splitmode
for run_name, accs_levels in results.items():
    splitmode = run_name.split("_")[0]

    for acc, level in accs_levels:
        splitmode_accuracies[splitmode][level].append(acc)

print(splitmode_accuracies)
# average the accuracies for each splitmode
for splitmode in splitmode_accuracies:
    for level in splitmode_accuracies[splitmode]:
        accuracies = splitmode_accuracies[splitmode][level]
        avg_accuracy = sum(accuracies) / len(accuracies)
        print("Avg Acc")
        print(f"{splitmode} {level}: {avg_accuracy:.4f}")
    
        
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