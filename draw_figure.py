# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('select_mlp.csv')
# # 提取数据
# rounds = df['Round']
# accuracies = df['Accuracy']
#
# # 创建图表
# plt.figure(figsize=(10, 5))
# plt.plot(rounds, accuracies, 'b-', label='Accuracy')
#
# # 添加标题和标签
# plt.title('MLP Select Model Accuracy over Rounds')
# plt.xlabel('Round')
# plt.ylabel('Accuracy')
#
# # 添加网格线
# plt.grid(True)
# # 添加图例
# plt.legend()
# # 显示图表
# plt.show()
# import re
# from medclip import constants
# client_ids = constants.CLIENT_IDS
# def extract_accuracy(file_path, client1, client2):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     # 使用正则表达式找到所有符合条件的模式
#     pattern = re.compile(rf'client model {client1} in {client2} data , its acc is (\d+\.\d+)')
#     matches = pattern.findall(content)
#     accuracies = [float(match) for match in matches]
#     acc = sum(accuracies) / len(accuracies)
#     print(f'{client1}model in {client2} data, acc is {acc}')
#
# # 指定你的txt文件路径
# file_path = 'log_fl.txt'
# for client1 in client_ids:
#     for client2 in client_ids:
#         extract_accuracy(file_path,client2,client1)
# import csv
#
#
# def convert_txt_to_csv(input_file, output_file):
#     # 打开输入的txt文件
#     with open(input_file, 'r') as txt_file:
#         lines = txt_file.readlines()
#
#     # 打开输出的csv文件
#     with open(output_file, 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#
#         # 写入csv文件的表头
#         csv_writer.writerow(['path', 'label', 'other'])
#
#         # 逐行解析txt文件的内容并写入csv文件
#         for line in lines:
#             parts = line.split()
#             path = parts[1]
#             label = parts[2]
#             other = parts[3]
#             csv_writer.writerow([path, label, other])
#
#
# # 指定输入和输出文件的路径
# input_file = 'test.txt'
# output_file = 'output.csv'
#
# # 调用转换函数
# convert_txt_to_csv(input_file, output_file)
# from transformers import AutoModel
# from PIL import Image
# import pandas as pd
#
# model = AutoModel.from_pretrained("./CXR-LLAVA-v2", trust_remote_code=True)
# model = model.to("cuda")
# import csv
#
#
# # 定义生成报告文本的函数
# def generate_report(path):
#     image_path = f'./images/{path}'
#     cxr_image = Image.open(image_path)
#     response = model.write_radiologic_report(cxr_image)
#     return response
#
# # 输入和输出文件名
# input_file = 'data.csv'  # 将 'your_file.csv' 替换为实际的CSV文件名
# output_file = 'generation.csv'  # 将 'your_file_with_reports.csv' 替换为你希望保存的文件名
# i = 0
# # 逐行读取和写入CSV文件
# with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
#         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
#     reader = csv.DictReader(infile)
#     fieldnames = reader.fieldnames
#
#     # 检查是否存在report列，如果不存在则添加
#     if 'report' not in fieldnames:
#         fieldnames.append('report')
#
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for row in reader:
#         if(row['labels']=='No Finding'):
#             continue
#         i += 1
#         print(i)
#         # 生成报告文本并添加到report列
#         row['report'] = generate_report(row['path'])
#         writer.writerow(row)
# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('data.csv')

# 要添加的新列名称
import csv
# input_file = 'b.csv'  # 将 'your_file.csv' 替换为实际的CSV文件名
# output_file = 'c.csv'  # 将 'your_file_with_reports.csv' 替换为你希望保存的文件名
# i = 0
#
# with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
#         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
#     reader = csv.DictReader(infile)
#     fieldnames = reader.fieldnames
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for row in reader:
#         labels = row['Labels'].split('|')
#         for label in labels:
#             if label == 'Nodule' or label == 'Emphysema' or label  == 'Fibrosis' or label == 'Hernia':
#                 continue
#             elif label == 'Pleural_Thickening':
#                 row['Pleural Other'] = 1
#             elif label == 'Mass':
#                 row['Lung Lesion'] = 1
#             elif label == 'Infiltration':
#                 row['Lung Opacity'] = 1
#             elif label == 'Effusion':
#                 row['Pleural Effusion'] = 1
#             elif label == 'Effusion':
#                 row['Pleural Effusion'] = 1
#             else:
#                 row[label] = 1
#         writer.writerow(row)



