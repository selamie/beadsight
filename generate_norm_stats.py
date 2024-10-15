from utils import get_norm_stats
import json


dataset_dir = '/home/selamg/beadsight/data/ssd/processed_ishape'

norm_stats = get_norm_stats(dataset_dir,num_episodes=96)
print(norm_stats)

for i in norm_stats.keys():
    norm_stats[i] = norm_stats[i].tolist()

with open('ishape_norm_stats.json','w') as f:
    json.dump(norm_stats,f,indent=4)


