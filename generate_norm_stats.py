from utils import get_norm_stats
import json


# dataset_dir = '/home/selamg/beadsight/data/ssd/processed_ishape'
dataset_dir = '/media/selamg/Crucial/selam/processed_drawer_supporting'
norm_stats = get_norm_stats(dataset_dir,num_episodes=100)
print(norm_stats)

for i in norm_stats.keys():
    norm_stats[i] = norm_stats[i].tolist()

with open('drawer_supporting_norm_stats.json','w') as f:
    json.dump(norm_stats,f,indent=4)


