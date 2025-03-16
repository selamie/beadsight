import json
import numpy as np
import matplotlib.pyplot as plt

# paths = {"Plugging": {"act": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_both_20",
#                                        "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/no_pretrain_both_20"},
#                               "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_vision_only_20",
#                                          "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/no_pretrain_vision_only_20"},
#                               "tactile": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_gel_only_20",
#                                           "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/no_pretrain_gel_only_20"}},
#                       "diffusion": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/pretrain_both_20",
#                                              "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/no_pretrain_both_20"},
#                                     "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/pretrain_vision_only_20",
#                                                "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/no_pretrain_vision_only_20"}}},
              
#          "cube_stacking": {"act": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/ACT/pretrain_both", 
#                                                   "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/ACT/no_pretrain_both"},
#                                     "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/ACT/pretrain_vision_only", 
#                                                "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/ACT/no_pretrain_vision_only"}},
#                             "diffusion": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/Diffusion/pretrain_both", 
#                                                    "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/Diffusion/no_pretrain_both"},
#                                           "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/Diffusion/pretrain_vision",
#                                                      "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Cube_stacking/Diffusion/no_pretrain_vision"}}},

#           "rectangle_stacking": {"act": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/ACT/pretrain_both", 
#                                                   "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/ACT/no_pretrain_both"},
#                                          "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/ACT/pretrain_vision_only",
#                                                     "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/ACT/no_pretrain_vision_only"}},
#                                  "diffusion": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/Diffusion/pretrain_both", 
#                                                         "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/Diffusion/no_pretrain_both"},
#                                                "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/Diffusion/pretrain_vision", 
#                                                           "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Rectangle_stacking/Diffusion/no_pretrain_vision"}}}}

paths = {"drawer":{"diffusion":{"vision": {"non-pretrained":"/home/selamg/beadsight/data/ssd/experiment_results/final/no_pretraining",
                                          "pretrained":"/home/selamg/beadsight/data/ssd/experiment_results/final/pretrained",
                                          "usb-pretrained":"/home/selamg/beadsight/data/ssd/experiment_results/final/usb_pretrained",
                                          "supporting-pretrained":"/home/selamg/beadsight/data/ssd/experiment_results/final/supporting_pretrained"}}}}


# success_rates = {"plugging": {"act": {"both": {"pretrained": 0.95, 
#                                                "non-pretrained": 0.9},
#                                       "vision": {"pretrained": 0.85, 
#                                                  "non-pretrained": 0.20},
#                                       "tactile": {"pretrained": 0.45, 
#                                                   "non-pretrained": 0.70}},

#                               "diffusion": {"both": {"pretrained": 0.75, 
#                                                      "non-pretrained": 0.70},
#                                             "vision": {"pretrained": 0.75, 
#                                                        "non-pretrained": 0.45}}},

#                 "cube_stacking": {"act": {"both": {"pretrained": 0.95, 
#                                                    "non-pretrained": 0.30},
#                                           "vision": {"pretrained": 1.00, 
#                                                      "non-pretrained": 0.95}},
        
#                                   "diffusion": {"both": {"pretrained": 0.7, 
#                                                          "non-pretrained": 0.55},
#                                                 "vision": {"pretrained": 0.95, 
#                                                            "non-pretrained": 0.75}}},

#                 "rectangle_stacking": {"act": {"both": {"pretrained": 0.80, 
#                                                         "non-pretrained": 0.70},
#                                                "vision": {"pretrained": 0.75, 
#                                                           "non-pretrained": 0.10}},
        
#                                        "diffusion": {"both": {"pretrained": 0.55, 
#                                                               "non-pretrained": 0.35},
#                                                      "vision": {"pretrained": 0.60, 
#                                                                 "non-pretrained": 0.35}}}}



def print_final_results():
    # print the results of the final trained policies for the fixed dataset using the ACT model:
    num_runs = 20
    for task in paths:
        for model in paths[task]:
            for camera_type in paths[task][model]:
                for pretrain in paths[task][model][camera_type]:
                    success = 0
                    for i in range(1, 1+num_runs):
                        run_stats_file = f"{paths[task][model][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                        # print(task, model, camera_type, pretrain, i)
                        # print(run_stats_file)
                        with open(run_stats_file, 'r') as f:
                            run_stats = json.load(f)
                        if run_stats["was_successful"] == "y":
                            success += 1
                        elif run_stats["was_successful"] == "n":
                            
                            pass
                        else:
                            raise ValueError(f"Unexpected value for was_successful: {run_stats['was_successful']}")
                    print(f"{task}: Model: {model}, Camera type: {camera_type}, Pretrain: {pretrain}, Success rate: {success}/{num_runs} ({100*success/num_runs:.2f}%)")
    # for camera_type in paths["fixed"]["act"]:
    #     for pretrain in paths["fixed"]["act"][camera_type]:
    #         success = 0
    #         for i in range(1, 1+num_runs):
    #             run_stats_file = f"{paths['fixed']['act'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
    #             with open(run_stats_file, 'r') as f:
    #                 run_stats = json.load(f)
    #             if run_stats["was_successful"] == "y":
    #                 success += 1
    #             elif run_stats["was_successful"] == "n":
    #                 pass
    #             else:
    #                 raise ValueError(f"Unexpected value for was_successful: {run_stats['was_successful']}")
    #         print(f"ACT: Camera type: {camera_type}, Pretrain: {pretrain}, Success rate: {success}/{num_runs} ({100*success/num_runs:.2f}%)")

    # for camera_type in paths["fixed"]["diffusion"]:
    #     for pretrain in paths["fixed"]["diffusion"][camera_type]:
    #         success = 0
    #         for i in range(1, 1+num_runs):
    #             run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
    #             with open(run_stats_file, 'r') as f:
    #                 run_stats = json.load(f)
    #             if run_stats["was_successful"] == "y":
    #                 success += 1
    #             elif run_stats["was_successful"] == "n":
    #                 pass
    #             else:
    #                 raise ValueError(f"Unexpected value for was_successful: {run_stats['was_successful']}")
    #         print(f"DIffusion: Camera type: {camera_type}, Pretrain: {pretrain}, Success rate: {success}/{num_runs} ({100*success/num_runs:.2f}%)")
            


def plot_percentiles():
    plt.figure()
    plt.title("Percentile Average Stress")
    num_runs = 20
    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            mean_strains = []
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                gelsight_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_strains.npy"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                strains = np.load(gelsight_file)
                closed_gripper_strains = strains[np.where(strains[:, 0] > 0.7)]
                # plt.plot(closed_gripper_strains[:, 1], label=f"1")
                # plt.plot(closed_gripper_strains[:, 2], label=f"2")
                # plt.legend()
                # plt.show()
                # mean_strains.append(np.mean(strains, axis=1))
                mean_strains.append(closed_gripper_strains[:, 2])
                # mean_strains = strains[:, 1]
            
            mean_strains = np.concatenate(mean_strains, axis=0)
            print('median for ', camera_type, pretrain, np.median(mean_strains))
            percentiles = np.percentile(mean_strains, range(101))
            plt.plot(percentiles, label=f"{camera_type}, {pretrain}")

    plt.legend()
    plt.show()

def plot_percentiles_diff_avg():
    plt.figure()
    plt.title("Percentile Average Stress")
    num_runs = 20
    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            mean_strains = []
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                gelsight_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_strains.npy"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                strains = np.load(gelsight_file)
                closed_gripper_strains = strains[np.where(strains[:, 0] > 0.7)]
                # plt.plot(closed_gripper_strains[:, 1], label=f"1")
                # plt.plot(closed_gripper_strains[:, 2], label=f"2")
                # plt.legend()
                # plt.show()
                # mean_strains.append(np.mean(strains, axis=1))
                mean_strains.append(closed_gripper_strains[:, 2])
                # mean_strains = strains[:, 1]
            
            mean_strains = np.concatenate(mean_strains, axis=0)
            percentiles = np.percentile(mean_strains, range(101))
            print('median for ', camera_type, pretrain, np.median(mean_strains))
            plt.plot(percentiles, label=f"{camera_type}, {pretrain}")

    plt.legend()
    plt.show()

def plot_percentiles_diffusion():
    num_runs = 20
    plt.figure()
    plt.title("Percentile Max Stress")
    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            max_strains = []
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                gelsight_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_max_strains.npy"
                gelsight_avg_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_strains.npy"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                strains = np.load(gelsight_file)
                avg_strains = np.load(gelsight_avg_file)
                max_strains.append(strains)
            
            max_strains = np.concatenate(max_strains, axis=0)
            percentiles = np.percentile(max_strains, range(101))
            plt.plot(percentiles, label=f"{camera_type}, {pretrain}")
    
    plt.legend()
    plt.show()

# make a bar chart of the success rate, with a blue bar for ACT and a green bar for Diffusion. 
# The bars should be grouped by camera type, with the pre-trained and non-pretrained models side by side.
# Pretrained models should be darker than non-pretrained models.
def plot_results():

    act_color = "lightblue"
    act_edge = "black"
    diff_color = "lightgreen"
    diff_edge = "black"
    pretrained_hatch = "//"
    non_pretrained_hatch = ""

    width = 0.2

    # plot the plugging graph
    fig, ax = plt.subplots(figsize=(5, 3))
    x = [0, 1, 2]

    # add x axis labels
    ax.set_xticks([x[0], x[1], x[2]-width])
    ax.set_xticklabels(["Tactile + Vision", "Vision Only", "Tactile Only"])

    # set y axis limits (so that the legend doesn't overlap with the bars)
    ax.set_ylim(0, 110)


    # set the y ticks to be in increments of 0.1 (max at 1)
    ax.set_yticks(np.arange(0, 120, 20))

    # add horizontal grid lines
    ax.yaxis.grid(True, zorder=0)

    # set the y axis label
    ax.set_ylabel("Success Rate (%)")


    # plugging
    act_both = [success_rates["plugging"]["act"]["both"]["pretrained"], success_rates["plugging"]["act"]["both"]["non-pretrained"]]
    diff_both = [success_rates["plugging"]["diffusion"]["both"]["pretrained"], success_rates["plugging"]["diffusion"]["both"]["non-pretrained"]]
    ax.bar(x[0] - 1.6*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor=act_edge, zorder=3)
    ax.bar(x[0] - 0.6*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor=act_edge, zorder=3)
    ax.bar(x[0] + 0.6*width, diff_both[1]*100, width, color=diff_color, hatch=non_pretrained_hatch, edgecolor=diff_edge, zorder=3)
    ax.bar(x[0] + 1.6*width, diff_both[0]*100, width, color=diff_color, hatch=pretrained_hatch, edgecolor=diff_edge, zorder=3)


    act_both = [success_rates["plugging"]["act"]["vision"]["pretrained"], success_rates["plugging"]["act"]["vision"]["non-pretrained"]]
    diff_both = [success_rates["plugging"]["diffusion"]["vision"]["pretrained"], success_rates["plugging"]["diffusion"]["vision"]["non-pretrained"]]
    ax.bar(x[1] - 1.6*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor='black', zorder=3)
    ax.bar(x[1] - 0.6*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor='black', zorder=3)
    ax.bar(x[1] + 0.6*width, diff_both[1]*100, width, color=diff_color, hatch=non_pretrained_hatch, edgecolor='black', zorder=3)
    ax.bar(x[1] + 1.6*width, diff_both[0]*100, width, color=diff_color, hatch=pretrained_hatch, edgecolor='black', zorder=3)
    
    act_both = [success_rates["plugging"]["act"]["tactile"]["pretrained"], success_rates["plugging"]["act"]["tactile"]["non-pretrained"]]
    ax.bar(x[2] - 1.5*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor=act_edge, zorder=3)
    ax.bar(x[2] - 0.5*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor=act_edge, zorder=3)

    # ax.legend(["ACT (not pretrained)", "ACT (pretrained)", "Diffusion (not pretrained)", "Diffusion (pretrained)"],
    #         loc = 'lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    
    plt.title("USB Plugging Task")
    plt.savefig("plugging.png", dpi=600, bbox_inches='tight')

    # plot the cube stacking graph and the rectangle stacking graph:
    for task_name in ["cube_stacking", "rectangle_stacking", "plugging"]:
        if task_name == "plugging":
            fig, ax = plt.subplots(figsize=(10, 3)) # use this to get the legend
        else:
            fig, ax = plt.subplots(figsize=(3.85, 3))
        x = [0, 1]

        # add x axis labels
        ax.set_xticks([x[0], x[1]])
        ax.set_xticklabels(["Tactile + Vision", "Vision Only"])

        # set y axis limits (so that the legend doesn't overlap with the bars)
        ax.set_ylim(0, 110)


        # set the y ticks to be in increments of 0.1 (max at 1)
        ax.set_yticks(np.arange(0, 120, 20))

        # add horizontal grid lines
        ax.yaxis.grid(True, zorder=0)

        # set the y axis label
        ax.set_ylabel("Success Rate (%)")

        act_both = [success_rates[task_name]["act"]["both"]["pretrained"], success_rates[task_name]["act"]["both"]["non-pretrained"]]
        diff_both = [success_rates[task_name]["diffusion"]["both"]["pretrained"], success_rates[task_name]["diffusion"]["both"]["non-pretrained"]]
        ax.bar(x[0] - 1.6*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor=act_edge, zorder=3)
        ax.bar(x[0] - 0.6*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor=act_edge, zorder=3)
        ax.bar(x[0] + 0.6*width, diff_both[1]*100, width, color=diff_color, hatch=non_pretrained_hatch, edgecolor=diff_edge, zorder=3)
        ax.bar(x[0] + 1.6*width, diff_both[0]*100, width, color=diff_color, hatch=pretrained_hatch, edgecolor=diff_edge, zorder=3)

        act_both = [success_rates[task_name]["act"]["vision"]["pretrained"], success_rates[task_name]["act"]["vision"]["non-pretrained"]]
        diff_both = [success_rates[task_name]["diffusion"]["vision"]["pretrained"], success_rates[task_name]["diffusion"]["vision"]["non-pretrained"]]
        ax.bar(x[1] - 1.6*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor='black', zorder=3)
        ax.bar(x[1] - 0.6*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor='black', zorder=3)
        ax.bar(x[1] + 0.6*width, diff_both[1]*100, width, color=diff_color, hatch=non_pretrained_hatch, edgecolor='black', zorder=3)
        ax.bar(x[1] + 1.6*width, diff_both[0]*100, width, color=diff_color, hatch=pretrained_hatch, edgecolor='black', zorder=3)

        if task_name == "plugging":
            ax.legend(["ACT (not pretrained)", "ACT (pretrained)", "Diffusion (not pretrained)", "Diffusion (pretrained)"],
                    loc = 'lower center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=False, ncol=4)

        if task_name == "cube_stacking":
            plt.title("Cube Stacking Task")
            plt.savefig("cube_stacking.png", dpi=600, bbox_inches='tight')

        elif task_name == "rectangle_stacking":
            plt.title("Rectangle Stacking Task")
            plt.savefig("rectangle_stacking.png", dpi=600, bbox_inches='tight')

        else:
            plt.title("Plugging Task")
            plt.savefig("legend.png", dpi=600, bbox_inches='tight')
        
    plt.show()


if __name__ == "__main__":
    print_final_results()
    # plot_results()
    # plot_percentiles_diffusion()
    # plot_percentiles()
    # plot_percentiles_diff_avg()


