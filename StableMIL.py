"""
作者：因吉 (Inki)
邮箱：inki.yinji@gmail.com
创建时间： 2021 0630
进一次修改：2021 0630.
注意：参考文献：Robust Multi-Instance Learning with Stable Instances (2020)
     参考博客：https://blog.csdn.net/weixin_44575152/article/details/118208353
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from Prototype import MIL
from FunctionTool import get_k_cross_validation_index, max_similarity


class StableMIL(MIL):

    def __init__(self, path, tau=0.25, k=10, bag_space=None):
        """
        构造器
        :param      tau: 阈值
        """
        super(StableMIL, self).__init__(path, bag_space=bag_space)
        self.tau = tau
        self.k = k
        self.vector = []
        self.tr_idx = []
        self.te_idx = []
        self.__initialize_stable_mil()

    def __initialize_stable_mil(self):
        """
        初始化
        """
        self.vector = np.zeros((self.num_bag, self.num_att))
        for i in range(self.num_bag):
            temp_bag = self.bag_space[i][0][:, :-1]
            self.vector[i] = np.average(temp_bag, 0)

    def __mapping(self, ins_pool):
        """
        包映射
        """
        ret_map = np.zeros((self.num_bag, len(ins_pool)))
        for i in range(self.num_bag):
            bag = self.bag_space[i][0][:, :-1]
            for j, ins in enumerate(ins_pool):
                ret_map[i][j] = max_similarity(bag, ins)
        return ret_map

    def get_mapping(self):
        """
        Split training set and test set.
        """

        # 获取训练集测试集索引
        self.tr_idx, self.te_idx = get_k_cross_validation_index(self.num_bag)
        # 找到负包标签
        negative_bag_lab = min(self.bag_lab)
        # 主循环
        for loop in range(self.k):
            # 步骤1. 使用simple-mil配对J48作为基分类器
            simple_mi_training_data = self.vector[self.tr_idx[loop]]
            simple_mi_training_lab = self.bag_lab[self.tr_idx[loop]]
            simple_mi_classifier = DecisionTreeClassifier()
            simple_mi_classifier.fit(simple_mi_training_data, simple_mi_training_lab)
            del simple_mi_training_data, simple_mi_training_lab

            # 步骤2. 找到因果实例池
            # 遍历正包中的每一个实例，时间复杂度着实不小，这也是选用simpl-MI算法的原因
            ins_pool, s_pool = [], []
            for i in self.tr_idx[loop][: min(50, len(self.tr_idx[loop]))]:
                if self.bag_lab[i] == negative_bag_lab:
                    continue
                bag_positive = self.bag_space[i][0][:, :-1]
                for ins in bag_positive:
                    bag_negative_new_list = []
                    # 遍历每一个负包
                    for j in self.tr_idx[loop]:
                        if self.bag_lab[j] != negative_bag_lab:
                            continue
                        bag_negative = self.bag_space[j, 0][:, :-1]
                        bag_negative_new = np.vstack([bag_negative, ins])
                        bag_negative_new = np.average(bag_negative_new, axis=0).tolist()
                        bag_negative_new_list.append(bag_negative_new)
                    bag_negative_new_list = np.array(bag_negative_new_list)
                    # 计算 s 值
                    predict_lab = simple_mi_classifier.predict(bag_negative_new_list)
                    s = np.sum(predict_lab) / len(predict_lab)
                    ins_pool.append(ins.tolist())
                    s_pool.append(s)

            # 找到满足阈值的实例
            ins_pool = np.array(ins_pool)
            s_max, s_min = np.max(s_pool), np.min(s_pool)
            tau = (s_max - s_min) * self.tau
            ins_idx = np.where(s_pool > tau)
            ins_pool = ins_pool[ins_idx]

            # 使用因果实例池完成包的映射
            mapping_mat = self.__mapping(ins_pool)

            yield (mapping_mat[self.tr_idx[loop]], self.bag_lab[self.tr_idx[loop]],
                   mapping_mat[self.te_idx[loop]], self.bag_lab[self.te_idx[loop]], None)
