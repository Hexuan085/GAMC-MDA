import subprocess
import re
import random
import pickle

# wmk =['00000_content', '00002_frontier_stitching', '00006_blackmarks', '00016_deepmarks', '00019_jia', '00156_deepsignwb_50bit']
wmk = ['00005_unrelated', '00000_noise']
wm_acc = {}
for wmk_index in wmk:
    wmk_current_acc ={}
    # 定义命令和参数
    command = "python"
    script_name = "steal.py"

    # 你感兴趣的变量名列表
    variables_of_interest = ['surrogate_test_acc_after_attack', 'surrogate_wm_acc_after_attack']

    # 执行命令10次并记录输出
    wm_acc[(wmk_index, variables_of_interest[0])] = []
    wm_acc[(wmk_index, variables_of_interest[1])] = []
    wmk_current_acc[variables_of_interest[0]]=[]
    wmk_current_acc[variables_of_interest[1]]=[]

    for i in range(50):
        m=random.sample([1,2,3], 1)
        print(m)
        args = [
            "--attack_config", 
            f"configs/cifar10/attack_configs/input_noising{m[0]}.yaml",
            "--wm_dir",
            f"/home/xhe085/Proj/NEW_WMK_copy/Watermark-Robustness-Toolbox/best_point/{wmk_index}"
        ]
        # 完整命令
        full_command = [command, script_name] + args
        # 运行命令并捕获输出
        result = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 打印和保存感兴趣的输出
        print(f"Execution {i+1}:")
        
        # 使用正则表达式从输出中提取感兴趣的变量值
        for var in variables_of_interest:
            # 正则表达式模式，匹配 "VariableName: Value" 形式的行
            pattern = f"{var}: (.+)"
            match = re.search(pattern, result.stdout)
            if match:
                value = match.group(1)
                print(f"{wmk_index}({var}): {value}")
                wm_acc[(wmk_index, var)].append(value)
                wmk_current_acc[var].append(value)
                # 将感兴趣的变量值保存到文件
                # with open(f"output_{i+1}.txt", "a") as f:  # 'a' 模式是为了追加到文件
                #     f.write(f"{var}: {value}\n")
            else:
                print(f"{var} not found in output.")
    print('wmk_current_acc', wmk_current_acc)
    with open(f'/home/xhe085/Proj/NEW_WMK_copy/Watermark-Robustness-Toolbox/outputs/cifar10/stability/{wmk_index}.pkl', 'wb') as file:
        pickle.dump(wmk_current_acc, file)