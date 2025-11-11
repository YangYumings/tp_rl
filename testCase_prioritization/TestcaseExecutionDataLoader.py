import pandas as pd

from ci_cycle import CICycleLog


class TestCaseExecutionDataLoader:
    complexity_metric_list = ["AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "AvgEssential",
                              "AvgLine",
                              "AvgLineBlank", "AvgLineCode", "AvgLineComment", "CountDeclClass", "CountDeclClassMethod",
                              "CountDeclClassVariable", "CountDeclExecutableUnit", "CountDeclFunction",
                              "CountDeclInstanceMethod",
                              "CountDeclInstanceVariable", "CountDeclMethod", "CountDeclMethodDefault",
                              "CountDeclMethodPrivate",
                              "CountDeclMethodProtected", "CountDeclMethodPublic", "CountLine", "CountLineBlank",
                              "CountLineCode",
                              "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment", "CountSemicolon",
                              "CountStmt",
                              "CountStmtDecl",
                              "CountStmtExe", "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict",
                              "MaxEssential",
                              "MaxNesting", "RatioCommentToCode", "SumCyclomatic", "SumCyclomaticModified",
                              "SumCyclomaticStrict", "SumEssential"]

    def __init__(self, data_path, data_format):
        self.data_path = data_path
        self.data_format = data_format
        self.test_data = None

    def load_data(self):
        last_results = []
        # CI 周期数
        cycle_ids = []
        max_size = 0
        # process last result
        if self.data_format == "simple":
            # pandas 读取CSV数据
            df = pd.read_csv(self.data_path, error_bad_lines=False, sep=",")
            # 遍历每一行
            for i in range(df.shape[0]):
                # 提出 LastResults 列，去除中括号，转为整数列表
                # 若为空，转为空列表
                last_result_str: str = df["LastResults"][i]
                temp_list = (last_result_str.strip("[").strip("]").split(","))
                if temp_list[0] != '':
                    last_results.append(list(map(int, temp_list)))
                else:
                    last_results.append([])
            df["LastResults"] = last_results
            self.test_data = df
        elif self.data_format == "enriched":
            df = pd.read_csv(self.data_path, error_bad_lines=False, sep=",")
            # df = df.rename(columns={'test_class_name': 'Id', 'time': 'last_exec_time',
            #                        'current_failures': 'Verdict'}, inplace=True)
            previous_cycle_commit = df["cycle_id"][0]
            cycle_id = 1
            for i in range(df.shape[0]):
                last_result = []
                last_result.append(df["failures_0"][i])
                last_result.append(df["failures_1"][i])
                last_result.append(df["failures_2"][i])
                last_result.append(df["failures_3"][i])
                last_results.append(last_result)
                if df["cycle_id"][i] != previous_cycle_commit:
                    assert len(df.loc[df['cycle_id'] == previous_cycle_commit]) == cycle_ids.count(cycle_id)
                    previous_cycle_commit = df["cycle_id"][i]
                    cycle_id = cycle_id + 1

                cycle_ids.append(cycle_id)

            df["LastResults"] = last_results
            df["Cycle"] = cycle_ids
            self.test_data = df
        return self.test_data

    def pre_process(self):
        # 查找数据中最小和最大的构建周期(Cycle)
        min_cycle = min(self.test_data["Cycle"])
        max_cycle = max(self.test_data["Cycle"])
        # 初始化空列表，用于存储处理后的CI周期日志
        ci_cycle_logs = []
        ### process all cycles and save them in a list of CiCycleLohs
        if self.data_format == 'simple':
            # 遍历所有周期ID，为每个周期创建一个 CICycleLog 实例，并将所有测试用例添加到该实例中
            for i in range(min_cycle, max_cycle + 1):
                ci_cycle_log = CICycleLog(i)
                cycle_rew_data = self.test_data.loc[self.test_data['Cycle'] == i]
                # 遍历每个测试用例，并将其添加到 CiCycleLog 实例中
                # ID、名称、执行时间、执行结果、失败历史、周期ID、执行时间分组、时间分组、执行时间历史
                for index, test_case in cycle_rew_data.iterrows():
                    ci_cycle_log.add_test_case(test_id=test_case["Id"], test_suite=test_case["Name"],
                                               avg_exec_time=test_case["Duration"],
                                               last_exec_time=test_case["Duration"],
                                               verdict=test_case["Verdict"],
                                               failure_history=test_case["LastResults"],
                                               cycle_id=test_case["Cycle"],
                                               duration_group=test_case["DurationGroup"],
                                               time_group=test_case["TimeGroup"],
                                               exec_time_history=None)

                # 将处理完的周期日志添加到列表中
                ci_cycle_logs.append(ci_cycle_log)
        elif self.data_format == 'enriched':
            for i in range(min_cycle, max_cycle + 1):
                ci_cycle_log = CICycleLog(i)
                cycle_rew_data = self.test_data.loc[self.test_data['Cycle'] == i]
                for index, test_case in cycle_rew_data.iterrows():
                    # add_test_case_enriched(self, test_id, test_suite, last_exec_time, verdict, avg_exec_time,
                    #                       failure_history=[], rest_hist=[], complexity=[]):
                    rest_hist = []
                    rest_hist.append(test_case["failures_%"])
                    rest_hist.append(test_case["time_since"])
                    rest_hist.append(test_case["tests"])
                    complexity_metrics = []
                    for metric in TestCaseExecutionDataLoader.complexity_metric_list:
                        complexity_metrics.append(test_case[metric])
                    ci_cycle_log.add_test_case_enriched(test_id=test_case["test_class_name"],
                                                        test_suite=test_case["test_class_name"],
                                                        last_exec_time=test_case["time"],
                                                        verdict=test_case["current_failures"],
                                                        avg_exec_time=test_case["time_0"],
                                                        failure_history=test_case["LastResults"],
                                                        rest_hist=rest_hist,
                                                        complexity_metrics=complexity_metrics,
                                                        cycle_id=test_case["cycle_id"])
                ci_cycle_logs.append(ci_cycle_log)

        return ci_cycle_logs
