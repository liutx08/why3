import os
import logging
import pandas as pd
import numpy as np
from itertools import islice
from functools import partial
from tqdm import tqdm as std_tqdm
from deepblock.api import APICVAEComplex

tqdm = partial(std_tqdm, dynamic_ncols=True)

CONSERVATIVE_SUCCESS_RATE = 0.5


class MolecularGenerator:
    def __init__(self, device="cpu"):
        """初始化 API """
        self.api = APICVAEComplex(device=device)

    def generate_molecules(self, input_data, input_type="pdb_id", num_samples=16, batch_size=8, random_seed=20230607):
        """
        生成分子
        :param input_data: 用户输入的数据 (PDB ID or file path)
        :param input_type: 输入数据的类型
        :param num_samples: 生成样本数量
        :param batch_size: 批处理大小
        :param random_seed: 随机种子
        :return: 生成的分子数据
        """
        # 计算 max_attempts
        conservative_max_attempts = int(np.ceil(num_samples / CONSERVATIVE_SUCCESS_RATE))
        max_attempts = conservative_max_attempts

        # 创建分子
        maker = self.api.item_make(input_data, input_type, None, None, 'sdf_fn')

        # 采样
        sampler = self.api.chem_sample(maker.item, batch_size=batch_size,
                                       max_attempts=max_attempts,
                                       desc=maker.item.id)

        pbar = tqdm(islice(sampler, num_samples), total=num_samples, desc="Generating Molecules")
        res_lst = [res for res in pbar]
        pbar.close()

        # 计算成功率
        succ_rate = sampler.num_success / sampler.num_attempts * 100
        logging.info(f'Success Rate: {succ_rate:.3f}%')

        # 评估
        evaluator = self.api.mol_evaluate(res.smi for res in res_lst)
        ind_lst = list(evaluator)

        # 返回结果
        output_obj = [{'smi': res.smi, 'frags': res.frags, 'ind': ind} for res, ind in zip(res_lst, ind_lst)]
        return output_obj

