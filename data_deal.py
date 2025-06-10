import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, window_size=3):
    """
    加载并预处理数据，生成滑动窗口数据
    参数:
    file_path (str): CSV文件路径
    window_size (int): 滑动窗口的大小
    返回:
    Tuple[np.ndarray, np.ndarray]: 返回输入和标签
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 去掉日期列，只保留流量数据
    traffic_data = data.drop(columns=['date'])

    # # 归一化数据
    # scaler = MinMaxScaler()
    # traffic_data = pd.DataFrame(scaler.fit_transform(traffic_data), columns=traffic_data.columns)

    def create_sliding_window_for_each_region(data, window_size):
        """
        生成滑动窗口数据
        参数:
        data (pd.DataFrame): 原始数据
        window_size (int): 滑动窗口的大小
        返回:
        Tuple[np.ndarray, np.ndarray]: 返回输入和标签
        """
        x, y = [], []

        # 对每一列（区域）进行处理
        for column in data.columns:
            region_data = data[column].values
            num_samples = len(region_data) - window_size

            for i in range(num_samples):
                x.append(region_data[i:i+window_size])
                y.append(region_data[i+window_size])

        return np.array(x), np.array(y)

    # 生成滑动窗口数据
    x, y = create_sliding_window_for_each_region(traffic_data, window_size)
    return x, y

def split_data(x, y, train_size=0.7, val_size=0.1, test_size=0.2):
    """
    分割数据为训练集、验证集和测试集
    参数:
    x (np.ndarray): 输入数据
    y (np.ndarray): 标签数据
    train_size (float): 训练集比例
    val_size (float): 验证集比例
    test_size (float): 测试集比例
    返回:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 分割后的数据集
    """
    # 确保比例总和为1
    assert train_size + val_size + test_size == 1.0

    # 分割数据为临时集和测试集（不打乱数据顺序）
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1 - train_size, random_state=42, shuffle=False)

    # 计算验证集和测试集比例
    val_test_ratio = val_size / (val_size + test_size)

    # 分割临时集为训练集和验证集（打乱数据顺序）
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1 - val_test_ratio, random_state=42, shuffle=True)

    # 对训练集进行归一化
    scaler_x = MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_val = scaler_x.transform(x_val)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)


    return x_train, x_val, x_test, y_train, y_val, y_test



# # 示例文件路径
# file_path = 'D:/yansdnu/traffic forecast/dataset/milano_traffic_nid.csv'  # 请将此路径替换为你的实际CSV文件路径
#
# # 加载并预处理数据
# window_size = 3
# x, y = load_and_preprocess_data(file_path, window_size)
#
# # 分割数据集
# train_size = 0.7
# val_size = 0.1
# test_size = 0.2
# x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, train_size, val_size, test_size)
# # x_train, x_val, x_test, y_train, y_val, y_test, scaler_x, scaler_y = split_data(x, y, train_size=0.7, val_size=0.1, test_size=0.2)
# # print("X scaler - Min:", scaler_x.data_min_, "Max:", scaler_x.data_max_)
# # print("Y scaler - Min:", scaler_y.data_min_, "Max:", scaler_y.data_max_)
#
# # 打印部分处理后的数据
# print("\nSample data from training set (x_train, y_train):")
# for i in range(3):
#     print(f"x_train[{i}]: {x_train[i]}, y_train[{i}]: {y_train[i]}")
#
# print("\nSample data from validation set (x_val, y_val):")
# for i in range(3):
#     print(f"x_val[{i}]: {x_val[i]}, y_val[{i}]: {y_val[i]}")
#
# print("\nSample data from test set (x_test, y_test):")
# for i in range(3):
#     print(f"x_test[{i}]: {x_test[i]}, y_test[{i}]: {y_test[i]}")