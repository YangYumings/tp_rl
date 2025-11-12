from typing import Any, Union

import numpy as np
import gym
from gym import spaces
from sklearn import preprocessing
from ci_cycle import CICycleLog
from Config import Config
import random

# 环境根据 智能体的动作 生成 奖励函数、观察集pair、测试用例排序结果
# 环境根据每个动作，将测试用例加入临时排序表
# 环境中能获取 每轮CI 的测试用例特征/向量集，因此每轮CI都需要重新创建一个新的 环境

class CIPairWiseEnv(gym.Env):
    # 初始化：一个是当前 CI周期 的测试用例特征集，Config 是模型与实验的配置信息
    def __init__(self, cycle_logs: CICycleLog, conf: Config):
        super(CIPairWiseEnv, self).__init__()
        self.conf = conf
        # self.db_type = conf.db_type
        self.reward_range = (-1, 1)
        self.cycle_logs = cycle_logs
        # 随机打乱测试用例顺序，保证每次训练时的测试用例初始状态不同
        random.shuffle(self.cycle_logs.test_cases)
        # 每个测试用例向量的长度 Test case 0 keys: ['test_id', 'test_suite', 'avg_exec_time', 'verdict', 'duration_group',
        # 'time_group', 'cycle_id', 'last_exec_time', 'failure_history', 'age']
        self.testcase_vector_size = self.cycle_logs.get_test_case_vector_length(cycle_logs.test_cases[0],
                                                                                self.conf.win_size)
        # TODO 此处的复制是为了观察，但观察的作用是什么？
        self.initial_observation = cycle_logs.test_cases.copy()
        self.test_cases_vector = self.initial_observation.copy()
        self.test_cases_vector_temp = []

        self.current_indexes = [0, 1]
        self.sorted_test_cases_vector = []
        self.current_obs = np.zeros((2, self.testcase_vector_size))
        self.width = 1
        self.right = 1
        self.left = 0
        self.end = 2
        self.index = 0
        self.current_indexes[0] = self.index
        self.current_indexes[1] = self.index + self.width
        self.current_obs = self.get_pair_data(self.current_indexes)  # 当前观察集

        # self.number_of_actions = len(self.cycle_logs.test_cases)
        # 两个动作：
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, self.testcase_vector_size))  # ID, execution time and LastResults

    # 返回当前测试用例向量
    def get_test_cases_vector(self):
        return self.test_cases_vector

    # 返回测试用例向量： current_indexes 指定 pair 中的测试用例索引的范围
    def get_pair_data(self, current_indexes):
        i = 0
        test_case_vector_length = \
            self.cycle_logs.get_test_case_vector_length(self.test_cases_vector[current_indexes[0]], self.conf.win_size)
        # 创建一个 2 行的矩阵，存储两个测试用例的向量
        temp_obs = np.zeros((2, test_case_vector_length))

        # 获取连个测试用例的特征向量
        for test_index in current_indexes:
            temp_obs[i, :] = self.cycle_logs.export_test_case(self.test_cases_vector[test_index],
                                                              "list_avg_exec_with_failed_history",
                                                              self.conf.padding_digit,  # 填充数字
                                                              self.conf.win_size)  # 历史窗口大小
            i = i + 1
        # 归一化：按列对特征矩阵进行最大值归一化处理
        temp_obs = preprocessing.normalize(temp_obs, axis=0, norm='max')
        return temp_obs

    def render(self, mode='human'):
        pass

    def reset(self):
        self.test_cases_vector = self.initial_observation.copy()
        self.current_indexes = [0, 1]
        self.width = 1
        self.right = 1
        self.left = 0
        self.end = 2
        self.index = 0
        self.current_obs = self.get_pair_data(self.current_indexes)
        self.test_cases_vector_temp = []
        return self.current_obs

    # TODO 下个观察值如何保证和上一个不同的
    def _next_observation(self, index):
        self.current_obs = self.get_pair_data(self.current_indexes)
        return self.current_obs

    def _initial_obs(self):
        random.shuffle(self.cycle_logs)  # 打乱测试用例顺序
        self.initial_observation = self.cycle_logs.test_cases.copy()  # 副本，作为初始观察状态
        self.test_cases_vector = self.initial_observation.copy()  # 副本，用于后续处理
        return self.initial_observation  # 返回初始观察状态

    ## the reward function must be called before updating the observation
    ## enriched dataset
    # 计算奖励值：参数是选择的哪个测试用例
    def _calculate_reward_enriched(self, test_case_index):
        if test_case_index == 0:
            selected_test_case = self.test_cases_vector[self.current_indexes[0]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[1]]
        else:
            selected_test_case = self.test_cases_vector[self.current_indexes[1]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[0]]
        # 如果选中的测试用例的测试结果比未选中的测试用例的测试结果好，则奖励值为1；否则奖励值为0；
        # 最后一次执行的时间早于为选中的测试用例，则奖励值为.5；否则奖励值为0；
        if selected_test_case['verdict'] > no_selected_test_case['verdict']:
            reward = 1
        elif selected_test_case['verdict'] < no_selected_test_case['verdict']:
            reward = 0
        elif selected_test_case['last_exec_time'] <= no_selected_test_case['last_exec_time']:
            reward = .5
        elif selected_test_case['last_exec_time'] > no_selected_test_case['last_exec_time']:
            reward = 0
        return reward

    ## simple  data set
    def _calculate_reward_simple(self, test_case_index):
        if test_case_index == 0:
            selected_test_case = self.test_cases_vector[self.current_indexes[0]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[1]]
        else:
            selected_test_case = self.test_cases_vector[self.current_indexes[1]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[0]]
        if selected_test_case['verdict'] > no_selected_test_case['verdict']:
            reward = 1
        elif selected_test_case['verdict'] < no_selected_test_case['verdict']:
            reward = 0
        elif selected_test_case['last_exec_time'] <= no_selected_test_case['last_exec_time']:
            reward = .5
        # 唯一的不同点
        elif selected_test_case['last_exec_time'] > no_selected_test_case['last_exec_time']:
            reward = 1
        return reward

    # 位置交换，通过交换实现测试用例的排序
    def swapPositions(self, l, pos1, pos2):
        l[pos1], l[pos2] = l[pos2], l[pos1]
        return l

    # 只有 last_exec 执行时间给的奖励函数不同
    def _calculate_reward(self, test_case_index):
        if self.conf.dataset_type == "simple":
            return self._calculate_reward_simple(test_case_index)
        elif self.conf.dataset_type == "enriched":
            return self._calculate_reward_enriched(test_case_index)
        else:
            assert False, "dataset type error"

    # 每次执行更新环境状态，返回下一个观察值、奖励、是否结束
    # test_case_index 当前动作（选择pair中的哪个测试用例）
    # 当所有测试用例都排序后，再增加的 Step 都是没有意义的
    def step(self, test_case_index):
        # 价值
        reward = self._calculate_reward(test_case_index)
        done = False
        # an step of a merging sort
        if test_case_index == 1:
            self.test_cases_vector_temp.append(self.test_cases_vector[self.right])
            self.right = self.right + 1
            if self.right >= min(self.end, self.index + 2 * self.width):
                while self.left < self.index + self.width:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
                    self.left = self.left + 1
        elif test_case_index == 0:
            self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
            self.left = self.left + 1
            if self.left >= self.index + self.width:
                while self.right < min(self.end, self.index + 2 * self.width):
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.right])
                    self.right = self.right + 1

        if self.right < self.end or self.left < self.index + self.width:
            None
        elif self.end < len(self.test_cases_vector):
            self.index = min(self.index + self.width * 2, len(self.test_cases_vector) - 1)
            self.left = self.index
            self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
            self.end = min(self.right + self.width, len(self.test_cases_vector))
            if self.right < self.left + self.width:
                while self.left < self.end:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
                    self.left = self.left + 1
                self.width = self.width * 2
                self.test_cases_vector = self.test_cases_vector_temp.copy()
                self.test_cases_vector_temp = []
                self.index = 0
                self.left = self.index
                self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
                self.end = min(self.right + self.width, len(self.test_cases_vector))
        elif self.width < len(self.test_cases_vector) / 2:
            self.width = self.width * 2
            self.test_cases_vector = self.test_cases_vector_temp.copy()
            self.test_cases_vector_temp = []
            self.index = 0
            self.left = self.index
            self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
            self.end = min(self.right + self.width, len(self.test_cases_vector))
        else:
            done = True
            ## a2c reset the env when the epsiode is done, so we need to copy the result of test cases
            self.test_cases_vector = self.test_cases_vector_temp.copy()
            assert len(self.test_cases_vector) == len(
                self.cycle_logs.test_cases), "merge sort does not work as expected"
            self.sorted_test_cases_vector = self.test_cases_vector.copy()
            return self.current_obs, reward, done, {}

        # TODO 更新 current_indexes
        if not done:
            self.current_indexes[0] = self.left
            self.current_indexes[1] = self.right
            self.current_obs = self._next_observation(test_case_index)
        return self.current_obs, reward, done, {}
