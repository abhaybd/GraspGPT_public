import csv
import os

res_folder = "/net/nfs2.prior/arijitr/research/semantic_grasping/GraspGPT_public/gcngrasp/results"
results_files = os.listdir("/net/nfs2.prior/arijitr/research/semantic_grasping/GraspGPT_public/gcngrasp/results")
results_files = [f for f in results_files if f.endswith(".csv")]

splitmode_accuracies = {}
for results_file in results_files:
    # open the results csv which is run_name and accuracy
    with open(os.path.join(res_folder, results_file), "r") as f:
        reader = csv.reader(f)
        results = {}
        for row in reader:
            run_name = row[0]
            accuracy = float(row[1].strip("%"))
            results[run_name] = accuracy

    # run name is splitmode_splitid. We want to average the accuracies for each splitmode
    
    for run_name, accuracy in results.items():
        splitmode = run_name.split("_")[0]
        if splitmode not in splitmode_accuracies:
            splitmode_accuracies[splitmode] = []
        splitmode_accuracies[splitmode].append(accuracy)

# average the accuracies for each splitmode
for splitmode, accuracies in splitmode_accuracies.items():
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"{splitmode}: {avg_accuracy:.4f}")

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