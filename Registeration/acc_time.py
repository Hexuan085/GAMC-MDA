import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

wmk = ['noise', 'unrelated', 'content', 'frontier_stitching', 'jia',  'blackmarks', 'deepmarks', 'deepsignwb']

data = {}
length = [2, 3, 4, 10, 20, 30]
length_mem = [10, 20, 30]
length_add = [2, 3, 4]
index = {2: 50, 3: 100, 4: 30}
for i in length:
    for j in wmk:
        with open(f'~/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/wm_png/{j}/wmk_dict_{i}.pkl', 'rb') as f:
            data[(i, j)] = pickle.load(f)

with open(f'~/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/metrcis_stability.pkl',
          'rb') as f:
    metrcis_stability = pickle.load(f)

with open(f'~/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/metrcis_stability_densenet.pkl',
          'rb') as f:
    metrcis_stability_densenet = pickle.load(f)

total_time = {}
sign_time = {}
single_time = {}
batch_time = {}
# 使用读取的数据
for i in length_mem:
    single_time[i] = []
    batch_time[i] = []
    for j in wmk:
        single_time[i].append(data[(i, j)]['acc_single_prove_mem_timing'])
        print(f'{i}, {j}', data[(i, j)]['acc_single_prove_mem_timing'])
        batch_time[i].append(data[(i, j)]['acc_batch_prove_mem_timing'])
        print(f'{i}, {j}', data[(i, j)]['acc_batch_prove_mem_timing'])

for i in length_add:
    total_time[index[i]] = []
    sign_time[index[i]] = []
    for j in wmk:
        total_time[index[i]].append(data[(i, j)]['acc_batch_add_genesis_timing'])
        print(f'{j}', data[(i, j)]['acc_batch_add_genesis_timing'])
        sign_time[index[i]].append(data[(i, j)]['acc_sign_timing'])
        print(f'{j}', data[(i, j)]['acc_sign_timing'])

schemes = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']
schemes_densenet = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']
# 假设每个子图的数据
data = {
    "(a) Accumulator vaule generation time": {
        "total_mem=30": total_time[30],
        "total_mem=50": total_time[50],
        "total_mem=100": total_time[100],
    },
    "(b) Total witness generation time by single method": {
        "wit_mem=10": single_time[10],
        "wit_mem=20": single_time[20],
        "wit_mem=30": single_time[30],
    },
    "(c) Total witness generation time by batch method": {
        "wit_mem=10": batch_time[10],
        "wit_mem=20": batch_time[20],
        "wit_mem=30": batch_time[30],
    }
}

line_styles = ['-', '--', '-.']
colors = ['blue', 'green', 'red']
line_widths = [1.5, 1.5, 1.5]

fig, axs = plt.subplots(1, 3, figsize=(18, 4))

for idx, (title, lengths) in enumerate(data.items()):
    for i, (length_label, times) in enumerate(lengths.items()):
        axs[idx].plot(schemes, times, label=length_label, color=colors[i], linestyle=line_styles[i], linewidth=line_widths[i])
    axs[idx].set_title(title)
    axs[idx].set_xlabel('Watermark scheme')
    axs[idx].set_ylabel('Time(s)')
    axs[idx].legend()

plt.setp(axs, xticks=schemes, xticklabels=schemes)
plt.setp(axs[0], yticks=[0, 0.06, 0.12, 0.18, 0.24, 0.3])
plt.setp(axs[1], yticks=[0, 1, 2, 3, 4, 5])
plt.setp(axs[2], yticks=[0, 0.03, 0.06, 0.12, 0.15, 0.18])
for ax in axs:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(10)

plt.tight_layout()
plt.show()

lengths = ['35%', '20%', '5%', '2%']

threshold = [0.02, 0.05, 0.20, 0.35]
wmk_stab =['00000_noise', '00005_unrelated', '00000_content', '00002_frontier_stitching', '00019_jia', '00006_blackmarks', '00016_deepmarks', '00156_deepsignwb_50bit']
wmk_stab_densenet =['00001_noise', '00003_unrelated', '00001_content', '00017_frontier_stitching', '00002_jia', '00000_blackmarks', '00008_deepmarks']

percent_t = {}
for t in threshold:
    percent_t[t] = []
    for w in wmk_stab:
        percent_t[t].append(metrcis_stability[(t,w)]*100)
# print(sign_time[30])

percent = np.array([
    percent_t[threshold[3]],
    percent_t[threshold[2]],
    percent_t[threshold[1]],
    percent_t[threshold[0]]
])

plt.figure(figsize=(12, 6))
sns.heatmap(percent, annot=True, fmt=".0f", xticklabels=schemes, yticklabels=lengths, cmap='Oranges')

plt.xlabel('(a) Watermark scheme for WRN',fontsize=14)
plt.ylabel('Threshold of decrease in WA',fontsize=14)
plt.title('Percentage of accurate identification under input noise',fontsize=14)

plt.xticks(rotation=45, fontsize=12)  
plt.yticks(rotation=90, fontsize=12)

# plt.tight_layout()
plt.show()

lengths = ['35%', '20%', '5%', '2%']

threshold = [0.02, 0.05, 0.20, 0.35]
wmk_stab_densenet =['00001_noise', '00003_unrelated', '00001_content', '00017_frontier_stitching', '00002_jia', '00000_blackmarks', '00008_deepmarks']

percent_t_densenet = {}
for t in threshold:
    percent_t_densenet[t] = []
    for w in wmk_stab_densenet:
        percent_t_densenet[t].append(metrcis_stability_densenet[(t, w)]*100)
# print(sign_time[30])
percent_densenet = np.array([
    percent_t_densenet[threshold[3]],
    percent_t_densenet[threshold[2]],
    percent_t_densenet[threshold[1]],
    percent_t_densenet[threshold[0]]
])

plt.figure(figsize=(12,6))
sns.heatmap(percent_densenet, annot=True, fmt=".0f", xticklabels=schemes_densenet, yticklabels=lengths, cmap='Oranges')

plt.xlabel('(b) Watermark scheme for Densenet',fontsize=14)
plt.ylabel('Threshold of decrease in WA',fontsize=14)

plt.xticks(rotation=45, fontsize=12)  # 旋转x轴标签以便于阅读，并设置字体大小
plt.yticks(rotation=90, fontsize=12)

# plt.tight_layout()
plt.show()

length_10_times = sign_time[30]
length_20_times = sign_time[50]
length_30_times = sign_time[100]

bar_width = 0.25
positions = np.arange(len(schemes))

plt.figure(figsize=(12, 6))
bars1 = plt.bar(positions - bar_width, length_10_times, bar_width, label='length=10')
bars2 = plt.bar(positions, length_20_times, bar_width, label='length=20')
bars3 = plt.bar(positions + bar_width, length_30_times, bar_width, label='length=30')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.legend()

plt.xticks(positions, schemes, fontsize=14)

plt.xlabel('wmk_schemes', fontsize=14)
plt.ylabel('time(s)', fontsize=14)

# plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()




