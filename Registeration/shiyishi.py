import matplotlib.pyplot as plt
import pickle

metric = ['max', 'min', 'mean']
length = [30, 50, 100]
with open(f'/Users/hexuan/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/wm_png/wmk_dict_100.pkl',
          'rb') as f:
    data = pickle.load(f)

line_styles = ['-', '-', '-']
schemes = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']
facecolor = ['#2F7FC1', '#F3D266', '#96C37D', '#C82423', '#9AC9DB', '#F8AC8C', '#FF8884']

edgecolor = ['#C82423', '#9AC9DB', '#F8AC8C']
j = 0
plt.figure(figsize=(8, 4))
bars = []
for i in length:

    plt.fill_between(schemes, data[('max', i)], data[('min', i)],  # 上限，下限
                     facecolor=facecolor[j],  # 填充颜色
                     edgecolor=edgecolor[j],  # 边界颜色
                     alpha=0.25)  # 透明度ps: // blog.csdn.net / OldDriver1995 / article / details / 116128063
    plt.plot(schemes, data[('max', i)],color=facecolor[j], linewidth=1, linestyle='--')
    plt.plot(schemes, data[('min', i)], color=facecolor[j], linewidth=1, linestyle='--')
    plt.plot(schemes, data[('mean', i)], color=facecolor[j], linestyle = line_styles[j], linewidth=3.5, label=f'length = {i}', marker='o',markersize=10)
    j = j + 1
plt.ylim(0, 30)
plt.legend(fontsize=15)
plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
plt.ylabel('Signature generation time(s)',fontsize=14)
plt.xlabel('Watermark schemes',fontsize=14)
plt.tight_layout()
plt.show()