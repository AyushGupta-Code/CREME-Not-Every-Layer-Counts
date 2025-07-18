#将humaneva_qwen目录下所有文件夹中的my子文件夹中的edit_results.csv放到外层文件夹中然后删除my文件夹
#如humaneva_qwen/A2/my/edit_results.csv -> humaneva_qwen/A2/edit_results.csv
#如humaneva_qwen/A3/my/edit_results.csv -> humaneva_qwen/A3/edit_results.csv
import os
import shutil

# # 定义根目录
# root_dir = "mbpp_qwen"

# # 遍历根目录下的所有文件夹
# for folder_name in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder_name)
#     print(f"处理文件夹: {folder_path}")
#     # 检查是否为文件夹
#     if os.path.isdir(folder_path):
#         my_folder_path = os.path.join(folder_path, "my")
#         edit_results_file = os.path.join(my_folder_path, "edit_result.csv")
#         # print(f"检查路径: {my_folder_path} 和文件: {edit_results_file}")
#         # print(os.path.exists(my_folder_path))
#         # print(os.path.isfile(edit_results_file))
#         # 检查 "my" 文件夹和 "edit_results.csv" 文件是否存在
#         if os.path.exists(my_folder_path) and os.path.isfile(edit_results_file):
#             # 目标路径
#             target_path = os.path.join(folder_path, "edit_results.csv")
            
#             # 移动文件
#             shutil.move(edit_results_file, target_path)
#             print(f"已移动: {edit_results_file} -> {target_path}")
            
#             # 删除 "my" 文件夹
#             shutil.rmtree(my_folder_path)
#             print(f"已删除文件夹: {my_folder_path}")


root_dir = "mbpp_qwen"
#遍历根目录下所有文件夹，将所有子文件夹下的edit_results.csv改名为edit_result.csv
# 遍历根目录下所有文件夹
for root, dirs, files in os.walk(root_dir):
    for file in files:
        # 检查文件名是否为edit_results.csv
        if file == "edit_results.csv":
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, "edit_result.csv")
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
