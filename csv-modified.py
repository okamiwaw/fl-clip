# import pandas as pd
#
# # 替换函数
# def replace_values(df, start_col, end_col):
#     columns = df.columns[start_col-1:end_col]
#     df[columns] = df[columns].replace({0: -1, -1: 0})
#     df[columns] = df[columns].fillna(0)
#     return df
#
# # 读取CSV文件
# file_path = 'raw2.csv'  # 替换为你的CSV文件路径
# df = pd.read_csv(file_path)
#
# # 请告诉我开始列和结束列的索引（i和j）
# start_col_index = 1 # 请在这里输入i的值
# end_col_index =  15# 请在这里输入j的值
#
# # 应用替换
# df = replace_values(df, start_col_index, end_col_index)
# # 读取CSV文件并清理
# output_file_path = 'b.csv'  # 您可以更改这个文件名
# df.to_csv(output_file_path, index=False)
# print("数据已成功保存到", output_file_path)

# import pandas as pd
#
# # 步骤1: 读取CSV文件
# df = pd.read_csv('d.csv')  # 替换为你的CSV文件路径
#
# # 步骤2: 分割DataFrame
# # 假设df的长度大于1000
# split_point = len(df) - 500  # 计算分割点
# df_front = df.iloc[:split_point]  # 前半部分
# df_back = df.iloc[split_point:]  # 后半部分（最后1000项）
#
# # 步骤3: 保存到新的CSV文件
# df_front.to_csv('e1.csv', index=False)  # 前半部分的CSV文件路径
# df_back.to_csv('e2.csv', index=False)  # 后半部分的CSV文件路径

# import pandas as pd
# import deepl
# import numpy as np
#
#
# def translate_chunk(chunk, translator, source_lang, target_lang):
#     """翻译DataFrame的一个小片段（chunk）"""
#     return chunk.apply(
#         lambda x: translator.translate_text(x, source_lang=source_lang, target_lang=target_lang).text if pd.notnull(
#             x) else x)
#
#
# def translate_column_in_csv(file_path, column_name, source_lang, target_lang, auth_key):
#     # 初始化DeepL翻译器
#     translator = deepl.Translator(auth_key)
#     # 读取整个CSV文件
#     df = pd.read_csv(file_path)
#
#     # 分批翻译：为了减少内存使用，我们使用numpy的array_split进行分块
#     n_chunks = np.ceil(len(df) / 50).astype(int)  # 确定需要多少个分块
#     chunks = np.array_split(df, n_chunks)  # 将DataFrame分割为多个块
#
#     for i, chunk in enumerate(chunks):
#         print(f"正在处理第 {i + 1} / {n_chunks} 批...")
#         # 翻译指定列
#         chunk[column_name] = translate_chunk(chunk[column_name], translator, source_lang, target_lang)
#         # 每个chunk处理后直接更新到df中，这里利用了chunk的引用性质
#         df.loc[chunk.index, column_name] = chunk[column_name]
#         # 将修改后的完整DataFrame写回原文件
#         df.to_csv(file_path, index=False)
#         print(f'第{i + 1} 块翻译完成，所有更改已保存。')
#         break
#
#
# if __name__ == "__main__":
#     translate_column_in_csv(
#         file_path='padchest.csv',  # 输入文件
#         column_name='Report',  # 需要翻译的列名
#         source_lang='ES',  # 源语言代码，西班牙语
#         target_lang='EN-US',  # 目标语言代码，英语
#         auth_key='7d315944-8028-4eb3-955a-e7ae5d267ad0:fx',  # 替换为你的DeepL API授权密钥
#     )



# def filter_rows_by_word_count(file_path, output_file_path, column_name):
#     """
#     Filters rows in a CSV file based on the word count in a specified column.
#
#     :param file_path: Path to the input CSV file.
#     :param output_file_path: Path where the filtered CSV file will be saved.
#     :param column_name: Name of the column to check for word count.
#     """
#     # Load the CSV file into a pandas DataFrame
#     df = pd.read_csv(file_path)
#
#     # Filter rows where the number of words in the specified column is at least 8
#     filtered_df = df[df[column_name].apply(lambda x: len(str(x).split()) >= 15)]
#
#     # Save the filtered DataFrame to a new CSV file
#     filtered_df.to_csv(output_file_path, index=False)
#     print(f"Filtered CSV has been saved to {output_file_path}")
#
#
# # Example usage
# file_path = 'raw.csv'  # Replace with your input file path
# output_file_path = 'filtered_output.csv'  # Desired path for the output file
# column_name = 'Report'  # Replace with the name of your column containing Spanish text
#
# filter_rows_by_word_count(file_path, output_file_path, column_name)


# import pandas as pd
# def remove_duplicate_rows(csv_file_path, output_file_path):
#     """
#     读取一个CSV文件，移除ImageID列中重复的行，然后保存到新的CSV文件。
#
#     参数:
#     - csv_file_path: 原CSV文件路径。
#     - output_file_path: 输出的CSV文件路径。
#     """
#     # 读取CSV文件
#     df = pd.read_csv(csv_file_path)
#
#     # 删除ImageID列中的重复行，保留第一次出现的行
#     df_unique = df.drop_duplicates(subset='Report')
#
#     # 将处理后的数据保存到新的CSV文件
#     df_unique.to_csv(output_file_path, index=False)
# # 示例用法
# csv_file_path = 'raw.csv'  # 替换为你的CSV文件路径
# output_file_path = 'output_file.csv'  # 替换为你想要的输出文件路径
# remove_duplicate_rows(csv_file_path, output_file_path)
# import pandas as pd
# # 加载CSV文件
# df = pd.read_csv('output_file.csv')
# # 过滤掉列中包含两个以上连续点的行
# filtered_df = df[~df['Report'].str.contains('\. \. +', regex=True)]
# # 将处理后的数据保存到新的CSV文件中
# filtered_df.to_csv('unique_filtered_data.csv', index=False)

#
# import pandas as pd
#
# # 加载CSV文件
# df = pd.read_csv('a.csv')
#
# # 初始化一个空的DataFrame来存储结果
# result_df = pd.DataFrame()
#
# # 对于第5到9列
# for col in range(6, 11):  # Python的索引从0开始，所以第5列实际上是索引4
#     # 找出该列值为1的行
#     filtered_rows = df[df.iloc[:, col] == 1]
#     # 如果找到的行数超过200，则只保留前200行
#     if len(filtered_rows) > 200:
#         filtered_rows = filtered_rows.iloc[:200]
#     # 将这些行添加到结果DataFrame
#     result_df = pd.concat([result_df, filtered_rows])
#
# # 保存结果DataFrame到一个新的CSV文件
# result_df.to_csv('c.csv', index=False)
#
# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('e.csv')
#
# df = df.iloc[:, 0:9]
#
#
# # 写入新的CSV文件
# df.to_csv('v.csv', index=False)
# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('c.csv')
#
# # 筛选条件：从第5列到第10列的绝对值和为1
# filtered_df = df[df.iloc[:, 5:10].sum(axis=1) == 1]
#
# filtered_df.to_csv('c.csv', index=False)


# import pandas as pd
#
# # 载入CSV文件
# df = pd.read_csv('raw.csv')
#
# # 检查文件行数，如果少于10,000行，则抽取全部行数
# n = min(15000, len(df))
#
# # 随机抽取10,000行
# sampled_df = df.sample(n)
#
# # 如果需要，可以将抽取的数据保存为新的CSV文件
# sampled_df.to_csv('sample_reports.csv', index=False)
# import pandas as pd
#
# # 读取CSV文件
# df_a = pd.read_csv('a.csv')
# df_b = pd.read_csv('b.csv')
#
# # 根据'report'列合并DataFrame以找出共有的行
# common_rows = pd.merge(df_a, df_b, on='Report')
#
# # 将结果保存到新的CSV文件
# common_rows.to_csv('common_rows.csv', index=False)
# import csv
#
# # 读取CSV文件
# with open('a.csv', 'r', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     data = list(reader)
#
# # 交换第三列和第四列（索引从0开始）
# for row in data:
#     row[0], row[6] = row[6], row[0]
#
# # 将修改后的数据写回CSV文件
# with open('b.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(data)
# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('client_4_v.csv')
#
# # 检查"report"列是否存在
#
# # 在"report"列的每个值前面加上'PadChest/'
# df['imgpath'] = 'PadChest/' + df['imgpath']
# # 保存修改后的数据到新的CSV文件中
# df.to_csv('modified_file.csv', index=False)
# import csv
#
# # 输入和输出文件路径
# input_file = 'client_4_t.csv'  # 这里替换为你的输入文件路径
# output_file = 'output.csv'  # 这里替换为你想要保存的输出文件路径
#
# # 读取原始CSV文件
# with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
#     reader = csv.reader(infile)
#     data = list(reader)
#
# # 添加ID列
# for index, row in enumerate(data, start=1):
#     row.insert(0, str(index))
#
# # 写入新的CSV文件
# with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
#     writer = csv.writer(outfile)
#     writer.writerows(data)

# import pandas as pd
# import numpy as np
#
# # 读取CSV文件
# df = pd.read_csv('client_1_t.csv')
#
# # 确保你的数据行数足够抽取5000项
# if len(df) < 6000:
#     raise ValueError("数据项不足6000，无法抽取。")
#
# # 随机抽取5000项
# random_indices = np.random.choice(df.index, size=5000, replace=False)
# sampled_df = df.loc[random_indices]
#
# # 保存到新的CSV文件
# sampled_df.to_csv('client_1_t.csv', index=False)
# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('a.csv')  # 替换 'your_file.csv' 为你的文件名
#
# # 添加新列，所有值设为1
# df['client'] = 3
#
# # 保存修改后的CSV文件
# df.to_csv('a.csv', index=False)  # 可以修改文件名为你希望的新文件名

# import pandas as pd
# import re
#
# # 读取CSV文件
# input_file = 'input.csv'
# output_file = 'output.csv'
# df = pd.read_csv(input_file)
#
# # 定义提取Chest后面的部分的函数
# def extract_chest_part(report):
#     # 使用正则表达式提取“chest”后的部分，不区分大小写
#     match1 = re.search(r'(?i)chest .*', report, re.DOTALL)
#     match2 = re.search(r'(?i)chest:.*', report, re.DOTALL)
#     match3 = re.search(r'[PERSONALNAME].*', report, re.DOTALL)
#     if match1:
#         return match1.group(0)
#     elif match2:
#         return match2.group(0)
#     elif match3:
#         return match3.group(0)
#     else:
#         return report  # 没有“chest”的情况保留原报告
#
# # 应用提取函数到report列
# df['Report'] = df['Report'].apply(extract_chest_part)
#
# # 保存处理后的数据到新的CSV文件
# df.to_csv(output_file, index=False)
#
# import pandas as pd
# import re
#
# # 读取CSV文件
# input_file = 'd.csv'
# output_file = 'e.csv'
# df = pd.read_csv(input_file)
#
# # 定义提取chest后面部分并去除transcribed后内容的函数（不区分大小写）
# def extract_chest_part(report):
#
#     cleaned_report = re.sub(r'Dr.*', '', report, flags=re.DOTALL)
#     return cleaned_report
#     # chest_match = re.search(r'Medical question:.*', report, re.DOTALL)
#     # if chest_match:
#     #     return chest_match.group(0)
#     # else:
#     #     return report  # 没有“chest”的情况保留原报告
#
# # 应用提取函数到report列
# df['Report'] = df['Report'].apply(extract_chest_part)
#
# # 保存处理后的数据到新的CSV文件
# df.to_csv(output_file, index=False)


# 定义一个函数来计算句子的单词数并过滤出单词数超过10的句子
# def filter_sentences(sentence):
#     # 使用/n分割句子
#     sentences = sentence.split('\n')
#     # 过滤出单词数超过10的句子
#     filtered = [s for s in sentences if 'ADDRESS' not in s]
#     # filtered = [s for s in sentences if 'COMMENT' not in s]
#     # filtered = [s for s in sentences if len(s.split()) >= 6]
#
#     # 返回过滤后的句子，重新用/n连接
#     return '\n'.join(filtered)
#
# # 应用函数到Report列
# df['Report'] = df['Report'].apply(filter_sentences)
# df = df[df['Report'] != '']
# # 输出或保存结果
# output_file_path = 'h.csv'
# df.to_csv(output_file_path, index=False)
#
# print("Filtered reports saved to:", output_file_path)
# import pandas as pd
#
# # 读取CSV文件
# df1 = pd.read_csv('b.csv')
# df2 = pd.read_csv('a.csv')
#
# # 按行对齐合并
# df_combined = pd.concat([df1, df2], axis=1)
#
# # 保存合并后的数据到新的CSV文件
# df_combined.to_csv('combined_file.csv', index=False)
#
# print("CSV文件已成功合并并保存为'combined_file.csv'")

# import pandas as pd
# import numpy as np
#
# # 读取CSV文件
# file_path = 'c.csv'  # 替换为你的CSV文件路径
# data = pd.read_csv(file_path)
#
# # 打乱数据
# shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # 计算分割点
# split_point = int(len(shuffled_data) * 2 / 3)
#
# # 拆分数据
# data_part1 = shuffled_data.iloc[:split_point]
# data_part2 = shuffled_data.iloc[split_point:]
#
# # 保存拆分后的数据
# data_part1.to_csv('part1.csv', index=False)
# data_part2.to_csv('part2.csv', index=False)
#
# print("CSV文件已成功拆分并保存为 part1.csv 和 part2.csv")
# import pandas as pd
# # 读取两个 CSV 文件
# df1 = pd.read_csv('client_1_v.csv')
# df2 = pd.read_csv('client_2_v.csv')
# df3 = pd.read_csv('client_3_v.csv')
# df4 = pd.read_csv('client_4_v.csv')
# # 合并两个数据框
# merged_df = pd.concat([df4, df1, df2, df3])
# # 保存合并后的数据框到新的 CSV 文件
# merged_df.to_csv('merged_file.csv', index=False)
# print("CSV files have been merged successfully into 'merged_file.csv'.")

import pandas as pd


df = pd.read_csv('client_4_t.csv')

# 在imgpath列的每个值前添加字符串'Candid/'
df['imgpath'] = df['imgpath'] + '.jpg'

# 将修改后的数据写回到新的CSV文件
df.to_csv('b.csv', index=False)

# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('client_4_test.csv')
#
# # 假设字符串列的名称是'string_column'
# # 去掉每个字符串的最后两个字符
# df['imgpath'] = df['imgpath'].apply(lambda x: x[:-2] if isinstance(x, str) else x)
#
# # 保存处理后的数据回CSV文件
# df.to_csv('client_4_test.csv', index=False)
#
# print("字符串处理完成，文件已保存为 'your_file_processed.csv'")


# import pandas as pd
#
# # 读取CSV文件
# file_path = 'client_4_test.csv'  # 替换为你的CSV文件路径
# df = pd.read_csv(file_path)
#
# # 假设需要修改的列名为 'column_name'
# column_name = 'imgpath'  # 替换为你的列名
#
# # 在每个字符串后面加上 .jpg
# df[column_name] = df[column_name].astype(str) + '.jpg'
#
# # 保存修改后的数据到新的CSV文件
# output_file_path = 'client_4_test.csv'  # 替换为你希望保存的文件路径
# df.to_csv(output_file_path, index=False)
#
# print(f"Modified CSV file has been saved to {output_file_path}")
