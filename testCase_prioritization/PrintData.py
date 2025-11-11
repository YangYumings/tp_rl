import pickle


# Cycle 351: 352, Test cases: 97
#   Test ID: 25498, Verdict: 0, Duration: 11110
#   Test ID: 25499, Verdict: 1, Duration: 29406
#   Test ID: 25500, Verdict: 0, Duration: 112842
#   Test ID: 25501, Verdict: 1, Duration: 29406
#   Test ID: 25502, Verdict: 1, Duration: 29406
def print_ci_cycle_logs(ci_cycle_logs):
    with open('ci_cycle_logs.pkl', 'wb') as f:
        pickle.dump(ci_cycle_logs, f)

    with open('ci_cycle_logs.pkl', 'rb') as f:
        loaded_logs = pickle.load(f)

    for i, cycle_log in enumerate(loaded_logs):
        print(f"Cycle {i}: {cycle_log.cycle_id}, Test cases: {cycle_log.get_test_cases_count()}")
        # 修改：检查 test_cases 是否存在且不为空，然后正确访问字典键
        if hasattr(cycle_log, 'test_cases') and cycle_log.test_cases:
            for test_case in cycle_log.test_cases:
                # 修改：使用字典键访问方式而不是属性访问方式
                if isinstance(test_case, dict):
                    print(f"  Test ID: {test_case.get('test_id', 'N/A')}, Verdict: {test_case.get('verdict', 'N/A')}, Duration: {test_case.get('last_exec_time', 'N/A')}")
                else:
                    # 如果 test_case 是对象而不是字典
                    print(f"  Test ID: {test_case.test_id}, Verdict: {test_case.verdict}, Duration: {test_case.last_exec_time}")
