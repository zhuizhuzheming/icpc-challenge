import sys
import math
import struct
import random

# ========== 常量定义 ==========
FP16_MAX = 65504.0          # IEEE-754 binary16 最大有限值
FP16_MIN_NORMAL = 6.103515625e-5  # 最小正规数（subnormal 通常 flush to zero）
FP32_MAX = 3.4028235e38     # IEEE-754 binary32 最大有限值

# ========== 安全 log10（防御性，虽题目保证安全） ==========
def safe_log10(x):
    return math.log10(abs(x) + math.pow(10, -301))

# ========== 基础模拟函数 ==========
def simulate_fp16_add(a: float, b: float) -> float:
    s = a + b
    if s == 0.0:
        return 0.0
    if not math.isfinite(s):
        return s
    try:
        bits = struct.unpack('>Q', struct.pack('>d', s))[0]
    except Exception:
        return s

    sign_bit = bits & (1 << 63)
    exp_bits = (bits >> 52) & 0x7FF
    mantissa_bits = bits & ((1 << 52) - 1)

    unbiased_exp = exp_bits - 1023
    if unbiased_exp < -14:
        return 0.0 if sign_bit == 0 else -0.0
    elif unbiased_exp > 15:
        return float('inf') if sign_bit == 0 else -float('inf')
    else:
        new_unbiased = max(-14, min(15, unbiased_exp))
        new_exp_bits = (new_unbiased + 1023) & 0x7FF
        new_bits = sign_bit | (new_exp_bits << 52) | mantissa_bits
        new_bits = new_bits & (~((1 << (52 - 10)) - 1))
        try:
            return struct.unpack('>d', struct.pack('>Q', new_bits))[0]
        except Exception:
            return s

def simulate_fp32_add(a: float, b: float) -> float:
    try:
        s = a + b
        return float(struct.unpack('>f', struct.pack('>f', float(s)))[0])
    except:
        return a + b

# ========== 安全性检查函数（新增） ==========
def is_safe_for_fp16_sum(values):
    if not values:
        return False
    if len(values) == 1:
        v = values[0]
        return v == 0.0 or (FP16_MIN_NORMAL <= abs(v) <= FP16_MAX)
    sum_abs = 0.0
    for v in values:
        av = abs(v)
        if av == 0.0:
            continue
        if av < FP16_MIN_NORMAL or av > FP16_MAX:
            return False
        sum_abs += av
        if sum_abs > FP16_MAX:
            return False
    return True

def is_safe_for_fp32_sum(values):
    if not values:
        return False
    if len(values) == 1:
        v = values[0]
        return v == 0.0 or (abs(v) <= FP32_MAX)
    sum_abs = 0.0
    for v in values:
        av = abs(v)
        if av == 0.0:
            continue
        if av > FP32_MAX:
            return False
        sum_abs += av
        if sum_abs > FP32_MAX:
            return False
    return True

def simulate_sum_with_precision(items, prec):
    s = 0.0
    for _, val in items:
        if prec == 'h':
            # 注意：这里仍用旧检查做兜底，但决策已由 is_safe_for_* 控制
            s = simulate_fp16_add(s, val)
            if not math.isfinite(s):
                return None
        elif prec == 's':
            s = simulate_fp32_add(s, val)
            if not math.isfinite(s):
                return None
        else:
            s += val
    return s

# ========== 策略网络（含稳健 update） ==========

class LinearPolicy:
    def __init__(self, input_dim=6, lr=0.001, global_ctx=None):
        self.input_dim = input_dim
        self.lr = lr
        bias_h, bias_s, bias_d = -1.0, 0.8, 0.2
        if global_ctx:
            if global_ctx.fp16_ratio > 0.8:
                bias_h += 1.5
            elif global_ctx.fp16_ratio < 0.3:
                bias_h -= 2.0
            if global_ctx.log_span > 15:
                bias_h -= 3.0
                bias_s -= 1.5
                bias_d += 2.0
        
        self.weights = []
        for action, base_bias in enumerate([bias_h, bias_s, bias_d]):
            w = [random.uniform(-0.1, 0.1) for _ in range(input_dim - 1)]
            w.append(base_bias + random.uniform(-0.2, 0.2))
            self.weights.append(w)

    def _softmax(self, logits):
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        s = sum(exps)
        return [e / s if s > 0 else 1/3 for e in exps]

    def choose_action(self, state, valid_mask):
        logits = [
            -1e9 if not valid_mask[a] else sum(self.weights[a][i] * state[i] for i in range(self.input_dim))
            for a in range(3)
        ]
        probs = self._softmax(logits)
        r = random.random()
        cum = 0.0
        for a, p in enumerate(probs):
            cum += p
            if r < cum:
                return a
        return 2

    def update(self, state, action, advantage):
        advantage = max(-0.5, min(0.5, advantage))
        for i in range(self.input_dim):
            self.weights[action][i] += self.lr * advantage * state[i]

# ========== 全局状态 ==========
class GlobalStateTracker:
    def __init__(self, indexed_values):
        self.true_total = sum(v for _, v in indexed_values)
        self.current_sim_sum = 0.0
        self.penalty_count = 0
        self.last_block_start = None

    def update_memory_penalty(self, indices):
        for i, idx in enumerate(indices):
            if i % 16 == 0:
                self.last_block_start = idx
            elif abs(idx - self.last_block_start) > 15:
                self.penalty_count += 1

class GlobalContext:
    def __init__(self, indexed_values):
        self.N = len(indexed_values)
        vals = [v for _, v in indexed_values if v != 0]
        if vals:
            logs = [safe_log10(v) for v in vals]
            self.log_span = max(max(logs) - min(logs), 0.0)
        else:
            self.log_span = 0.0
        # 使用新标准计算 fp16_ratio
        self.fp16_ratio = sum(
            1 for _, v in indexed_values 
            if (v == 0.0 or (abs(v) >= FP16_MIN_NORMAL and abs(v) <= FP16_MAX))
        ) / max(self.N, 1)

# ========== 分块 + 聚类 ==========
def adaptive_grouping(indexed_values, base_span=16, max_span=32):
    n = len(indexed_values)
    blocks = []
    for i in range(0, n, base_span):
        chunk = indexed_values[i:i+base_span]
        logs = [math.floor(safe_log10(v)) for _, v in chunk if v != 0]
        if not logs:
            main_log = -300
        else:
            main_log = max(set(logs), key=logs.count)
        blocks.append({'chunk': chunk, 'main_log': main_log, 'start': chunk[0][0], 'end': chunk[-1][0] if chunk else chunk[0][0]})
    
    merged = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        j = i + 1
        while j < len(blocks) and blocks[j]['main_log'] == cur['main_log'] and blocks[j]['end'] - cur['start'] < max_span:
            cur['chunk'].extend(blocks[j]['chunk'])
            cur['end'] = blocks[j]['end']
            j += 1
        merged.append(cur['chunk'])
        i = j
    return merged

# ========== 决策函数 ==========
def choose_precision(items, global_tracker, policy, global_N, remaining_n, global_ctx):
    true_sum = sum(v for _, v in items)
    logs = [math.floor(safe_log10(v)) for _, v in items if v != 0]
    log_span = min(max(logs) - min(logs), 10) if len(set(logs)) > 1 else 0

    state = [
        len(items) / 1000.0,
        log_span / 10.0,
        global_tracker.penalty_count / 1000.0,
        remaining_n / max(global_N, 1),
        global_ctx.fp16_ratio,
        1.0
    ]

    vals_only = [v for _, v in items]
    valid_mask = [
        is_safe_for_fp16_sum(vals_only),
        is_safe_for_fp32_sum(vals_only),
        True
    ]
    if abs(true_sum) / (abs(global_tracker.true_total) + 1e-20) > 0.1:
        valid_mask[0] = valid_mask[1] = False

    best_score = -1e9
    best_action = 2
    P = global_tracker.penalty_count / 20000.0

    for a, prec in enumerate(['h', 's', 'd']):
        if not valid_mask[a]:
            continue
        sim_sum = simulate_sum_with_precision(items, prec)
        if sim_sum is None or math.isnan(sim_sum) or math.isinf(sim_sum):
            continue

        abs_err = abs(true_sum - sim_sum)
        numerator = max(abs_err, 1e-20)
        denominator = max(abs(true_sum), 1e-200)
        A = (numerator / denominator) ** 0.05

        W = {'h':1, 's':2, 'd':4}[prec] * (len(items)-1)
        C = (W + P) / max(global_N - 1, 1)
        D = 10.0 / math.sqrt(C + 0.5)
        score = D / A

        if score > best_score:
            best_score = score
            best_action = a

    double_sum = simulate_sum_with_precision(items, 'd')
    if double_sum is not None and math.isfinite(double_sum):
        abs_err_d = abs(true_sum - double_sum)
        num_d = max(abs_err_d, 1e-20)
        den_d = max(abs(true_sum), 1e-200)
        A_d = (num_d / den_d) ** 0.05
        W_d = 4 * (len(items)-1)
        C_d = (W_d + P) / max(global_N - 1, 1)
        D_d = 10.0 / math.sqrt(C_d + 0.5)
        score_d = D_d / A_d
        advantage = (best_score - score_d) / (score_d + 1e-12)
    else:
        advantage = 0.0

    if len(items) >= 3 and policy is not None:
        policy.update(state, best_action, advantage)

    return ['h', 's', 'd'][best_action]

# ========== 递归求和 ==========
def recursive_sum(items, remaining_n, tracker, policy, N, ctx, n_ary=4):
    if len(items) <= 1:
        return str(items[0][0]) if items else "{d:1}"
    if len(items) <= n_ary:
        sorted_by_idx = sorted(items, key=lambda x: x[0])
        final_order = []
        for i in range(0, len(sorted_by_idx), 16):
            block = sorted_by_idx[i:i+16]
            block.sort(key=lambda x: abs(x[1]))
            final_order.extend(block)
        prec = choose_precision(final_order, tracker, policy, N, remaining_n, ctx)
        tracker.update_memory_penalty([idx for idx, _ in final_order])
        sim_val = simulate_sum_with_precision(final_order, prec)
        if sim_val is not None and math.isfinite(sim_val):
            tracker.current_sim_sum += sim_val
        return f"{{{prec}:{','.join(str(i) for i, _ in final_order)}}}"
    else:
        groups = [items[i:i+n_ary] for i in range(0, len(items), n_ary)]
        subs = [recursive_sum(g, remaining_n - len(g), tracker, policy, N, ctx, n_ary) for g in groups]
        return "{d:" + ",".join(subs) + "}"

# ========== 主逻辑 ==========
def solve():
    data = sys.stdin.read().split()
    if not data:
        print("{d:1}")
        return
    try:
        n = int(data[0])
        if n < 1:
            print("{d:1}")
            return
        values = list(map(float, data[1:1+n]))
        if len(values) != n:
            print("{d:1}")
            return
    except Exception:
        print("{d:1}")
        return

    indexed = [(i+1, values[i]) for i in range(n)]
    if n == 1:
        print("{d:1}")
        return

    tracker = GlobalStateTracker(indexed)
    ctx = GlobalContext(indexed)
    policy = LinearPolicy(input_dim=6, lr=0.001, global_ctx=ctx)

    groups = adaptive_grouping(indexed)
    
    sub_exprs = []
    remaining = n
    for group in groups:
        sorted_group = sorted(group, key=lambda x: x[0])
        buckets = {}
        for idx, val in sorted_group:
            k = -300 if val == 0 else math.floor(safe_log10(val))
            buckets.setdefault(k, []).append((idx, val))
        group_parts = []
        for log_key in sorted(buckets.keys(), reverse=True):
            bucket = buckets[log_key]
            remaining -= len(bucket)
            if len(bucket) == 1:
                group_parts.append(str(bucket[0][0]))
            else:
                expr = recursive_sum(bucket, remaining, tracker, policy, n, ctx, n_ary=4)
                group_parts.append(expr)
        if len(group_parts) == 1:
            sub_exprs.append(group_parts[0])
        else:
            sub_exprs.append("{d:" + ",".join(group_parts) + "}")
    
    final = sub_exprs[0] if len(sub_exprs) == 1 else "{d:" + ",".join(sub_exprs) + "}"
    print(final)

if __name__ == "__main__":

    solve()
