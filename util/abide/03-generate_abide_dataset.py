import deepdish as dd
import os.path as osp
import os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

def main(args):
    # 使用 args.root_path 作为根路径
    root_path = args.root_path
    data_dir = os.path.join(root_path, 'ABIDE_pcp/cpac/filt_noglobal/raw')
    timeseires = os.path.join(root_path, 'ABIDE_pcp/cpac/filt_noglobal')

    # 初始化数据列表
    times = []
    labels = []
    pcorrs = []
    corrs = []

    for f in os.listdir(data_dir):
        if osp.isfile(osp.join(data_dir, f)):
            fname = f.split('.')[0]

            try:
                # 加载时间序列数据
                files = os.listdir(osp.join(timeseires, fname))
                file = list(filter(lambda x: x.endswith("1D"), files))[0]
                time = np.loadtxt(osp.join(timeseires, fname, file), skiprows=0)

                # 如果是一维数组，重塑为 (140, 400)
                if time.ndim == 1:
                    expected_size = 140 * 400
                    if time.size != expected_size:
                        print(f"文件 {file} 数据大小不正确，期望 {expected_size}，实际 {time.size}")
                        continue
                    time = time.reshape(140, 400)

                # 转置为 (400, 140)
                time = time.T

                # 过滤时间序列长度
                if time.shape[1] < 100:
                    print(f"文件 {file} 时间序列长度不足 100，实际为 {time.shape[1]}，跳过")
                    continue

                # 加载 .h5 文件中的数据
                temp = dd.io.load(osp.join(data_dir, f))
                pcorr = temp['pcorr'][()]
                pcorr[pcorr == float('inf')] = 0

                att = temp['corr'][()]
                att[att == float('inf')] = 0

                label = temp['label']

                # 存储数据
                times.append(time[:,:100])
                labels.append(label[0])
                corrs.append(att)
                pcorrs.append(pcorr)

            except Exception as e:
                print(f"处理文件 {f} 失败: {str(e)}")
                continue

    # 保存数据
    np.save(os.path.join(root_path, 'ABIDE_pcp/abide.npy'), {
        'timeseires': np.array(times),
        "label": np.array(labels),
        "corr": np.array(corrs),
        "pcorr": np.array(pcorrs)
    })
    print(f"数据已保存至: {os.path.join(root_path, 'ABIDE_pcp/abide.npy')}")
    print(f"总共处理了 {len(labels)} 个受试者")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the final dataset')
    parser.add_argument('--root_path', default='../data', type=str, help='The path of the folder containing the dataset folder.')
    args = parser.parse_args()
    main(args)