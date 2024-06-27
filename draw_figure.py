# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # df = pd.read_csv('select_mlp.csv')
# # # 提取数据
# # rounds = df['Round']
# # accuracies = df['Accuracy']
# #
# # # 创建图表
# # plt.figure(figsize=(10, 5))
# # plt.plot(rounds, accuracies, 'b-', label='Accuracy')
# #
# # # 添加标题和标签
# # plt.title('MLP Select Model Accuracy over Rounds')
# # plt.xlabel('Round')
# # plt.ylabel('Accuracy')
# #
# # # 添加网格线
# # plt.grid(True)
# # # 添加图例
# # plt.legend()
# # # 显示图表
# # plt.show()
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
