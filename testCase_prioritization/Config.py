

class Config:
    def __init__(self):
        # 填补数据，数据预处理阶段填充缺失值
        self.padding_digit = -1
        # 时间窗口大小，后续默认为 10
        self.win_size = -1
        # 数据集类型。根据特征分为 simple、 complex
        self.dataset_type = "simple"
        # 最大测试用例数量
        self.max_test_cases_count = 400
        # 训练步数
        self.training_steps = 10000
        # 强化学习中的折扣因子，用于未来奖励的计算
        self.discount_factor = 0.9
        # 经验回放标志
        self.experience_replay = False
        # 训练的起始周期数
        self.first_cycle = 1
        # 周期总数,默认训练中 end 是第100个周期
        self.cycle_count = 100
        # 数据集
        self.train_data = "../data/tc_data_paintcontrol.csv"
        # 日志文件输出路径
        self.output_path = '../data/DQNAgent'
        # 日志文件名
        self.log_file="log.csv"