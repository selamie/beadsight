import json
import os
correct = 0
total = 0
for i in range(0, 41):
    file = f"/home/selamg/beadsight/data/ssd/experiment_results/ACT/no_pretrain_vision_only/run_{i}/run_stats.json"
    if not os.path.exists(file):
        continue
    with open(file, 'rb') as f:
        data = json.load(f)
    
    print(i, data['was_successful'])
    total += 1

    if data['was_successful'] == "y":
        correct += 1

print(correct/total)
print(total)