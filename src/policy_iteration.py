import numpy as np
import copy

# MDPの設定
# p[0]: p(s'=Office|s=Home, a=move)
# p[1]: p(s'=Bar|s=Office, a=move)
# p[2]: p(s'=Home|s=Bar, a=move)
p = [0.8, 0.5, 1.0]

# 割引率の設定
gamma = 0.95

# 報酬期待値の設定
# r[s, s', a]
# s ← 0: Home 1: Office 2: Bar
# a ← 0: move 1: stay
r = np.zeros((3, 3, 2))
r[0, 1, 0] = 1.0
r[0, 2, 0] = 2.0
r[0, 0, 1] = 0.0
r[1, 0, 0] = 1.0
r[1, 2, 0] = 2.0
r[1, 1, 1] = 1.0
r[2, 0, 0] = 1.0
r[2, 1, 0] = 0.0
r[2, 2, 1] = -1.0

# 価値関数の初期化
v = [0, 0, 0]
v_prev = copy.copy(v)

# 行動価値関数の初期化
# q[s, a]
q = np.zeros((3, 2))

# 方策分布の初期化
# pi[s]   ← (a=move)
pi = [0.5, 0.5, 0.5]


# 方策評価関数の定義
def policy_estimator(pi, p, r, gamma):
    """policy_estimator

    方策反復法

    Args:
        pi ([type]): 方策
        p ([type]): 遷移条件付き確率
        r ([type]): 報酬期待値
        gamma ([type]): 割引率

    Returns:
        [type]: 価値関数
    """
    # 初期化
    # 報酬ベクトル
    R = [0, 0, 0]
    # 状態遷移確率
    # P[s, s'] ← sからs'に移動する確率
    P = np.zeros((3, 3))
    # 計算途中の格納
    A = np.zeros((3, 3))

    for i in range(3):
        # 状態遷移列の計算
        # stay
        P[i, i] = 1 - pi[i]
        # move(0→1 or 1→2 or 2→0)
        P[i, (i + 1) % 3] = p[i] * pi[i]
        # move(0→2 or 1→0 or 2→1)
        P[i, (i + 2) % 3] = (1 - p[i]) * pi[i]

        # 報酬ベクトルの計算
        # R = sum_a{pi(a|s)} * sum_s'{p(s'|s, a) * r(s, a, s')}
        # 移動 + 移動しない
        R[i] = pi[i] * (p[i] * r[i, (i + 1) % 3, 0] + (1 - p[i]) * r[i, (i + 2) % 3, 0]) \
            + (1 - pi[i]) * r[i, i, 1]

    # 行列演算によるベルマン方程式の求解
    # np.eye(3) → [[1, 0, 0], [0, 1, 0], [0, 0, 1]] → 単位行列
    # v = (1 - gamma * P)^{-1} R
    A = np.eye(3) - gamma * P
    B = np.linalg.inv(A)  # 逆行列
    v_sol = np.dot(B, R)

    return v_sol


# 方策反復法の計算
for step in range(100):
    # 方策評価ステップ
    v = policy_estimator(pi, p, r, gamma)

    # 価値関数vが前ステップの値v_prepを改善しなければ終了
    if np.min(v - v_prev) <= 0:
        break

    # 現ステップの価値関数と方策を表示
    print('step: {} value: {} policy: {}'.format(step, v, pi))

    # 方策改善ステップ
    for i in range(3):
        # 行動価値関数を計算
        # q(s, a) = sum_{s'}p(s'|s, a)[r(s, a, s') + gamma * v(s')]
        # 移動
        q[i, 0] = p[i] * (r[i, (i + 1) % 3, 0] + gamma * v[(i + 1) % 3]) + (1 - p[i]) * (r[i, (i + 2) % 3, 0] + gamma * v[(i + 2) % 3])
        # 移動しない
        q[i, 1] = r[i, i, 1] + gamma * v[i]

        # 行動価値関数の元でgreedyに方策を改善
        # 移動するほうが行動価値が高い場合、
        if q[i, 0] > q[i, 1]:
            pi[i] = 1.0
        # 行動価値が同じ場合、
        elif q[i, 0] == q[i, 1]:
            pi[i] = 0.5
        # 行動しないほうが行動価値が高い場合、
        else:
            pi[i] = 0.0
    # 現ステップの価値関数を記録
    v_prev = copy.copy(v)
