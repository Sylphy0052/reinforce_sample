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
v_new = copy.copy(v)

# 行動価値関数の初期化
# q[s, a]
q = np.zeros((3, 2))

# 方策分布の初期化
# pi[s]   ← (a=move)
pi = [0.5, 0.5, 0.5]

# 方策反復法の計算
for step in range(1000):
    for i in range(3):
        # 行動価値関数の計算
        # a=move
        q[i, 0] = p[i] * (r[i, (i + 1) % 3, 0] + gamma * v[(i + 1) % 3]) \
            + (1 - p[i]) * (r[i, (i + 2) % 3, 0] + gamma * v[(i + 2) % 3])
        # a=stay
        q[i, 1] = r[i, i, 1] * gamma * v[i]

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

    # 改善された方策のもとで価値関数を計算
    # max_a{q(s, a)}
    v_new = np.max(q, axis=-1)

    # 価値関数vが前ステップの値v_prepを改善しなければ終了
    if np.min(v_new - v) <= 0:
        break

    # 価値関数を更新
    v = copy.copy(v_new)

    # 現ステップの価値関数と方策を表示
    print('step: {} value: {} policy: {}'.format(step, v, pi))
