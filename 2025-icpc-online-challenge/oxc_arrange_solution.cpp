/* 
 * OXC 光交换网络流量调度与连接配置优化算法
 * ==========================================
 * 
 * 本算法旨在解决大规模 OXC（光交叉连接器）网络中，多组服务器间通信流的高效、无冲突路径分配问题。
 * 核心目标是：在满足物理拓扑约束的前提下，最大化网络吞吐量、最小化资源竞争（冲突）、并均衡负载。
 * 
 * 算法采用"分而治之 + 智能协同"的混合策略，主要包含以下四个核心阶段：
 * 
 * 1. 【轻量级 MCF (Minimum Cost Flow) 引导】
 *    - 目标：为每个流量对（源组-目的组）分配平面的概率分布，实现全局负载均衡
 *    - 步骤：
 *       1.1 初始化每个组的平面高度（heights）和流量分配（flow_alloc）
 *       1.2 迭代更新：
 *           - 计算每个平面的总流量和拥塞程度
 *           - 根据流出/流入流量差和拥塞更新组平面高度
 *           - 使用softmax将高度差转换为流量分配概率
 *       1.3 输出每个流量对的平面权重矩阵，用于后续概率引导
 * 
 * 2. 【冲突分析与智能分组】
 *    - 目标：识别高冲突流量对并将其分组，实现分而治之
 *    - 步骤：
 *       2.1 拓扑感知冲突分析：
 *           - 计算流之间的组级别冲突（相同源组/目的组）
 *           - 新增拓扑感知计分器，考虑OXC连接复杂度和平面亲和度
 *           - 结合收敛比因子调整冲突权重
 *       2.2 分组策略：
 *           - 基于冲突阈值将流量分组，组内冲突最小化
 *           - 每组大小限制在MAX_GROUP_SIZE以内
 *           - 按组大小降序排列，优先处理大组
 * 
 * 3. 【基于神经网络策略微调的强化学习规划】
 *    - 目标：为每个组内的流量智能选择路径，优化局部和全局目标
 *    - 步骤：
 *       3.1 状态特征提取（6维）：
 *           - 剩余待分配流数量
 *           - 平均Spine负载
 *           - OXC端口使用率
 *           - 源组和目的组的历史使用模式（熵）
 *           - 流冲突度估计
 *           - 平面负载均衡度
 *       3.2 动作特征提取（5维）：
 *           - 路径负载（标准化）
 *           - 平面使用历史
 *           - OXC端口占用率
 *           - 调整成本估计
 *           - Spine负载均衡度
 *       3.3 轻量级神经网络架构：
 *           - 使用标准库实现的全连接神经网络
 *           - Xavier初始化，Leaky ReLU激活
 *           - 包含策略网络（Q网络）和价值网络（V网络）
 *       3.4 策略微调机制：
 *           - 基于论文方法的在线策略微调
 *           - 经验回放缓冲区存储状态-动作-奖励序列
 *           - TD学习更新网络参数
 *           - ε-greedy探索策略
 *       3.5 多步规划：
 *           - 使用蒙特卡洛rollout评估动作的长期回报
 *           - 规划时域为RL_PLANNING_HORIZON
 *       3.6 组内优化流程：
 *           - 第一阶段：基于MCF引导的随机搜索尝试分配所有流量
 *           - 第二阶段：对未分配的流量使用策略微调进行协同规划
 *           - 第三阶段：智能兜底和强制分配确保所有流量都有路由
 * 
 * 4. 【鲁棒性兜底与最终验证】
 *    - 目标：确保所有流量都有有效路由，保证解的可行性
 *    - 步骤：
 *       4.1 终极兜底机制：
 *           - 当策略微调无法找到有效路径时，强制建立连接
 *           - 允许清除现有连接以建立新连接
 *       4.2 全局验证：
 *           - 验证所有流量的路由有效性
 *           - 确保OXC连接的双向一致性
 *           - 修复不一致的连接状态
 *       4.3 输出准备：
 *           - 生成每个OXC的端口连接关系矩阵
 *           - 生成每个流量的完整路由路径
 * 
 * 算法特点：
 * - 拓扑感知：考虑网络收敛比和平面划分，动态调整冲突权重
 * - 在线学习：在规划过程中实时更新神经网络策略
 * - 分层优化：MCF引导全局负载均衡，分组优化局部冲突避免
 * - 鲁棒性强：多重兜底机制确保解的可行性
 * 
 * 参数说明：
 * - ALPHA, BETA: 评分函数权重
 * - MAX_GROUP_SIZE: 每组最大流量数
 * - CONFLICT_THRESHOLD: 冲突分组阈值
 * - MAX_LOAD_THRESHOLD: 最大负载阈值
 * - MCF_ITERATIONS: MCF引导迭代次数
 * - RL_*: 强化学习相关参数
 * 
 * 数据结构：
 * - Topology: 网络拓扑结构，包含N,S,L,M,K,P等参数
 * - ConflictAnalyzer: 冲突分析和分组器
 * - LightweightMCFGuide: 轻量级MCF引导器
 * - PolicyFineTuner: 策略微调器
 * - ImprovedGroupOptimizer: 改进的组优化器
 * - BatchOptimizer: 批量优化主控制器
 */

#include <bits/stdc++.h>
using namespace std;

// ==================== 常量定义 ====================
const double ALPHA = 1000.0;
const double BETA = 300.0;
const int MAX_GROUP_SIZE = 25;
const double CONFLICT_THRESHOLD = 8.0;
const int MAX_LOAD_THRESHOLD = 12;

// MCF引导相关常量
const int MCF_ITERATIONS = 3;
const double MCF_LEARNING_RATE = 0.1;
const double EPSILON = 1e-9;

// 神经网络策略微调相关常量
const double RL_LEARNING_RATE = 0.0005;
const double RL_DISCOUNT = 0.925;
const double RL_EPSILON = 0.2;
const int RL_STATE_DIM = 6;    // 状态特征维度
const int RL_ACTION_DIM = 5;   // 动作特征维度
const int RL_HIDDEN_DIM = 32;  // 隐藏层大小
const int RL_PLANNING_HORIZON = 3;  // 规划时域
const int RL_MAX_BUFFER_SIZE = 1000; // 经验缓冲区大小
const int RL_BATCH_SIZE = 64;  // 批量大小

// ==================== 随机数引擎 ====================
thread_local mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count()%998244353);

// ==================== 拓扑结构 ====================
struct Topology {
    int N, S, L, M, K, P;
    int SpinesPerPlane, OXCsPerPlane, R;
    double convergence_ratio;  // 收敛比因子
    
    Topology(int _N, int _S, int _L, int _M, int _K, int _P)
        : N(_N), S(_S), L(_L), M(_M), K(_K), P(_P) {
        SpinesPerPlane = S / P;
        OXCsPerPlane = M / P;
        R = N * SpinesPerPlane * K;
        
        // 计算收敛比因子 (M/P * K) / L
        double up_links = (M / P) * K;
        double down_links = L;
        double ratio = down_links / up_links;  // Down/Up 比例
        
        // 根据收敛比设置不同的冲突权重因子
        if (abs(ratio - 7.0) < 0.01) {
            convergence_ratio = 0.3;  // 1:7 收敛比，冲突可能性最小
        } else if (abs(ratio - 3.0) < 0.01) {
            convergence_ratio = 0.6;  // 1:3 收敛比
        } else {
            convergence_ratio = 1.0;  // 1:1 收敛比，冲突可能性最大
        }
    }
    
    inline int get_port(int oxc_id, int group, int spine_id, int k) const {
        int spine_local = spine_id % SpinesPerPlane;
        return group * SpinesPerPlane * K + spine_local * K + k;
    }
    
    tuple<int, int, int> decode_port(int port) const {
        if (port < 0 || port >= R) return {-1, -1, -1};
        int group = port / (SpinesPerPlane * K);
        int rem = port % (SpinesPerPlane * K);
        int spine_local = rem / K;
        int k = rem % K;
        return {group, spine_local, k};
    }
    
    int get_global_spine(int oxc_id, int spine_local) const {
        int plane = oxc_id / OXCsPerPlane;
        return plane * SpinesPerPlane + spine_local;
    }
};

// ==================== 工具函数：计算 Spine 负载 ====================
vector<vector<int>> compute_spine_load(const Topology& topo, const vector<vector<int>>& conn) {
    vector<vector<int>> spine_load(topo.S, vector<int>(topo.M, 0));
    for (int m = 0; m < topo.M; ++m) {
        for (int p = 0; p < topo.R; ++p) {
            if (conn[m][p] != -1 && p < conn[m][p]) {
                auto [gA, slA, kA] = topo.decode_port(p);
                auto [gB, slB, kB] = topo.decode_port(conn[m][p]);
                if (gA == -1 || gB == -1) continue;
                int sA = topo.get_global_spine(m, slA);
                int sB = topo.get_global_spine(m, slB);
                if (sA < topo.S && sB < topo.S) {
                    spine_load[sA][m]++;
                    spine_load[sB][m]++;
                }
            }
        }
    }
    return spine_load;
}

// ====================【改进版】冲突分析与分组（拓扑感知）====================
class ConflictAnalyzer {
    const vector<tuple<int, int, int, int>>& flows;
    double convergence_factor;
    
    // --- 新增：拓扑感知计分器 ---
    class TopologyAwareScorer {
        const Topology& topo;
        vector<double> oxc_complexity_weight; // 每个OXC的静态复杂度权重
        vector<vector<double>> plane_affinity_matrix; // 平面亲和度矩阵 [group][plane]
        
    public:
        TopologyAwareScorer(const Topology& t) : topo(t) {
            // 1. 计算每个OXC的连接复杂度（核心：连接的(Group, Spine)对越多，权重越高）
            oxc_complexity_weight.assign(topo.M, 0.0);
            for (int m = 0; m < topo.M; ++m) {
                int connections = 0;
                // 简单估算：该OXC在每个Group中，都连接到 SpinesPerPlane 个Spine
                connections = topo.N * topo.SpinesPerPlane;
                // 权重归一化：连接数除以最大可能连接数（所有组所有Spine）
                oxc_complexity_weight[m] = static_cast<double>(connections) / (topo.N * topo.S);
            }
            
            // 2. 初始化平面亲和度矩阵（鼓励同平面通信）
            plane_affinity_matrix.assign(topo.N, vector<double>(topo.P, 1.0)); 
            // 初始值为1.0，表示无额外亲和或惩罚
        }
        
        // 评估两条流若共用相同路径元素（如相同OXC）时的额外冲突风险
        double evaluate_topology_conflict(size_t i, size_t j, 
                                         const vector<tuple<int, int, int, int>>& flows,
                                         double convergence_factor) const {
            auto [gA1, lA1, gB1, lB1] = flows[i];
            auto [gA2, lA2, gB2, lB2] = flows[j];
            
            double topology_penalty = 0.0;
            
            // 关键冲突场景1：如果两条流可能竞争同一个OXC（组对相同或重叠）
            // 这是一种简化但有效的估计：若流共享同一个源组或目的组，它们极有可能使用相同的OXC平面
            bool potential_same_oxc = (gA1 == gA2) || (gA1 == gB2) || (gB1 == gA2) || (gB1 == gB2);
            
            if (potential_same_oxc) {
                // 为简单起见，我们取两个流可能使用的"逻辑中心"OXC平面
                // 实际算法中，这里可以替换为更精确的、基于MCF权重的OXC选择概率
                int assumed_plane_1 = (gA1 + gB1) % topo.P; // 示例：简单哈希
                int assumed_plane_2 = (gA2 + gB2) % topo.P;
                
                if (assumed_plane_1 == assumed_plane_2) {
                    // 如果它们可能落在同一平面，进一步检查该平面内的OXC负载
                    int sample_oxc = assumed_plane_1 * topo.OXCsPerPlane; // 取该平面第一个OXC为例
                    topology_penalty += 15.0 * oxc_complexity_weight[sample_oxc];
                }
            }
            
            // 关键冲突场景2：相同Leaf的流（你的原始关注点），结合收敛比
            // 此处我们保留你的原始逻辑，但用收敛比因子进行调制
            double leaf_penalty = 0.0;
            if (lA1 == lA2 || lA1 == lB2) leaf_penalty += 5.0;
            if (lB1 == lB2 || lB1 == lA2) leaf_penalty += 5.0;
            
            topology_penalty += leaf_penalty * convergence_factor;
            
            return topology_penalty;
        }
        
        // 提供一个接口，用于在路径规划后更新状态（可选，为未来动态调整预留）
        void update_after_assignment(int oxc_id, int group) {
            // 此处可实现在分配后更新OXC负载或平面亲和度
            // 例如：plane_affinity_matrix[group][oxc_id / topo.OXCsPerPlane] += 0.1;
        }
    };
    
    // --- 成员变量 ---
    TopologyAwareScorer scorer;
    
public:
    // 构造函数：接收流数据、收敛比和拓扑结构
    ConflictAnalyzer(const vector<tuple<int, int, int, int>>& f, double conv_ratio, const Topology& topo) 
        : flows(f), convergence_factor(conv_ratio), scorer(topo) {}
    
    // 主冲突度计算函数（集成拓扑感知）
    double conflict(size_t i, size_t j) const {
        double s = 0;
        
        // 1. 原有的组级别冲突（强惩罚）
        auto [gA1, lA1, gB1, lB1] = flows[i];
        auto [gA2, lA2, gB2, lB2] = flows[j];
        
        if (gA1 == gA2 || gA1 == gB2) s += 10;
        if (gB1 == gB2 || gB1 == gA2) s += 10;
        
        // 2. 新增：调用拓扑感知计分器
        s += scorer.evaluate_topology_conflict(i, j, flows, convergence_factor);
        
        // 3. 完全相同的通信对（最高惩罚）
        if ((gA1 == gA2 && lA1 == lA2 && gB1 == gB2 && lB1 == lB2) ||
            (gA1 == gB2 && lA1 == lB2 && gB1 == gA2 && lB1 == lA2)) {
            s += 25; // 略微提高，因为拓扑冲突已部分涵盖
        }
        
        return s;
    }
    
    // 分组逻辑保持不变
    vector<vector<size_t>> group(double threshold) const {
        vector<vector<size_t>> gs;
        vector<char> used(flows.size(), 0);
        
        for (size_t i = 0; i < flows.size(); ++i) {
            if (used[i]) continue;
            vector<size_t> g = {i};
            used[i] = 1;
            
            for (size_t j = i + 1; j < flows.size(); ++j) {
                if (used[j]) continue;
                double mx = 0;
                for (size_t m : g) mx = max(mx, conflict(m, j));
                if (mx < threshold) {
                    g.push_back(j);
                    used[j] = 1;
                    if ((int)g.size() >= MAX_GROUP_SIZE) break;
                }
            }
            gs.push_back(g);
        }
        
        sort(gs.begin(), gs.end(), [](const auto& a, const auto& b) {
            return a.size() > b.size();
        });
        return gs;
    }
};

// ==================== 路径合法性检查 ====================
bool is_path_valid(
    const Topology& topo,
    const vector<vector<int>>& oxc_conn,
    int gA, int /*lA*/, int gB, int /*lB*/,
    int spineA, int kA, int oxc, int spineB, int kB) {
    
    if (gA == gB) return false;
    
    int pA = topo.get_port(oxc, gA, spineA, kA);
    int pB = topo.get_port(oxc, gB, spineB, kB);
    
    if (pA == pB || pA >= topo.R || pB >= topo.R) return false;
    
    const auto& conn = oxc_conn[oxc];
    if (conn[pA] != -1 && conn[pA] != pB) return false;
    if (conn[pB] != -1 && conn[pB] != pA) return false;
    
    return true;
}

// ==================== 建立连接辅助函数 ====================
bool establish_connection(
    const Topology& topo,
    vector<vector<int>>& conn,
    int gA, int lA, int gB, int lB,
    int spineA, int kA, int oxc, int spineB, int kB,
    vector<vector<int>>& group_plane_usage,
    array<int, 5>& route) {
    
    if (!is_path_valid(topo, conn, gA, lA, gB, lB, spineA, kA, oxc, spineB, kB)) {
        return false;
    }
    
    int pA = topo.get_port(oxc, gA, spineA, kA);
    int pB = topo.get_port(oxc, gB, spineB, kB);
    
    if (pA < 0 || pA >= topo.R || pB < 0 || pB >= topo.R || pA == pB) {
        return false;
    }
    
    if (conn[oxc][pA] != -1 && conn[oxc][pA] != pB) {
        return false;
    }
    if (conn[oxc][pB] != -1 && conn[oxc][pB] != pA) {
        return false;
    }
    
    conn[oxc][pA] = pB;
    conn[oxc][pB] = pA;
    
    route = {spineA, kA, oxc, spineB, kB};
    
    int plane_id = oxc / topo.OXCsPerPlane;
    group_plane_usage[gA][plane_id]++;
    group_plane_usage[gB][plane_id]++;
    
    return true;
}

// ==================== 轻量级 MCF 引导器 ====================
class LightweightMCFGuide {
    int num_groups;
    int num_planes;
    int num_flows;
    vector<vector<double>> heights;
    vector<vector<double>> flow_alloc;
    vector<double> plane_capacity;
    vector<tuple<int, int>> flow_endpoints;

public:
    LightweightMCFGuide(const Topology& topo, const vector<tuple<int, int, int, int>>& flows)
        : num_groups(topo.N), num_planes(topo.P), num_flows(flows.size()) {
        
        heights.assign(num_groups, vector<double>(num_planes, 0.0));
        flow_alloc.assign(num_flows, vector<double>(num_planes, 1.0 / num_planes));
        plane_capacity.assign(num_planes, topo.SpinesPerPlane * topo.K);
        
        flow_endpoints.reserve(num_flows);
        for (auto [gA, lA, gB, lB] : flows) {
            flow_endpoints.emplace_back(gA, gB);
        }
    }

    vector<vector<double>> run() {
        for (int iter = 0; iter < MCF_ITERATIONS; ++iter) {
            vector<double> total_flow(num_planes, 0.0);
            for (int f = 0; f < num_flows; ++f) {
                for (int p = 0; p < num_planes; ++p) {
                    total_flow[p] += flow_alloc[f][p];
                }
            }
            
            vector<double> congestion(num_planes);
            for (int p = 0; p < num_planes; ++p) {
                congestion[p] = max(0.0, total_flow[p] - plane_capacity[p]);
            }

            vector<vector<double>> new_heights = heights;
            for (int g = 0; g < num_groups; ++g) {
                for (int p = 0; p < num_planes; ++p) {
                    double outflow = 0.0, inflow = 0.0;
                    for (int f = 0; f < num_flows; ++f) {
                        auto [src, dst] = flow_endpoints[f];
                        if (src == g) outflow += flow_alloc[f][p];
                        if (dst == g) inflow += flow_alloc[f][p];
                    }
                    new_heights[g][p] += MCF_LEARNING_RATE * (outflow - inflow - congestion[p]);
                }
            }
            heights = move(new_heights);

            for (int f = 0; f < num_flows; ++f) {
                auto [src, dst] = flow_endpoints[f];
                vector<double> gradients(num_planes);
                double max_grad = -1e18;
                for (int p = 0; p < num_planes; ++p) {
                    gradients[p] = heights[dst][p] - heights[src][p];
                    max_grad = max(max_grad, gradients[p]);
                }
                
                double sum_exp = 0.0;
                for (int p = 0; p < num_planes; ++p) {
                    double exp_val = exp(- (gradients[p] - max_grad));
                    sum_exp += exp_val;
                    flow_alloc[f][p] = exp_val;
                }
                
                for (int p = 0; p < num_planes; ++p) {
                    flow_alloc[f][p] /= (sum_exp + EPSILON);
                }
            }
        }

        vector<vector<double>> weights(num_flows, vector<double>(num_planes, 0.0));
        for (int f = 0; f < num_flows; ++f) {
            double sum_inv = 0.0;
            for (int p = 0; p < num_planes; ++p) {
                weights[f][p] = 1.0 / (flow_alloc[f][p] + EPSILON);
                sum_inv += weights[f][p];
            }
            for (int p = 0; p < num_planes; ++p) {
                weights[f][p] /= (sum_inv + EPSILON);
            }
        }
        return weights;
    }
};

// ==================== 状态特征提取器（保持与原代码兼容）====================
vector<double> extract_state_features(
    const Topology& topo,
    int gA, int gB,
    const vector<vector<int>>& spine_load,
    const vector<vector<int>>& oxc_conn,
    const vector<vector<int>>& group_plane_usage,
    int remaining_flows) {
    
    vector<double> features(RL_STATE_DIM, 0.0);
    
    // 特征1: 剩余待分配流数量（标准化）
    features[0] = remaining_flows / 100.0;
    
    // 特征2: 平均Spine负载
    double avg_spine_load = 0.0;
    int spine_count = 0;
    for (int s = 0; s < topo.S; ++s) {
        for (int m = 0; m < topo.M; ++m) {
            avg_spine_load += spine_load[s][m];
            spine_count++;
        }
    }
    if (spine_count > 0) features[1] = avg_spine_load / spine_count / MAX_LOAD_THRESHOLD;
    
    // 特征3: OXC端口使用率
    int total_ports = topo.M * topo.R;
    int used_ports = 0;
    for (int m = 0; m < topo.M; ++m) {
        for (int p = 0; p < topo.R; ++p) {
            if (oxc_conn[m][p] != -1 && p < oxc_conn[m][p]) {
                used_ports += 2;
            }
        }
    }
    features[2] = static_cast<double>(used_ports) / total_ports;
    
    // 特征4: 源组和目的组的历史使用模式
    int total_hist = 0;
    for (int p = 0; p < topo.P; ++p) {
        total_hist += group_plane_usage[gA][p] + group_plane_usage[gB][p];
    }
    if (total_hist > 0) {
        double entropy = 0.0;
        for (int p = 0; p < topo.P; ++p) {
            double prob = (group_plane_usage[gA][p] + group_plane_usage[gB][p]) / static_cast<double>(total_hist);
            if (prob > 0) entropy -= prob * log(prob + 1e-10);
        }
        features[3] = entropy / log(topo.P + 1e-10);
    }
    
    // 特征5: 流冲突度估计
    int pair_usage = 0;
    for (int p = 0; p < topo.P; ++p) {
        pair_usage += min(group_plane_usage[gA][p], group_plane_usage[gB][p]);
    }
    features[4] = min(1.0, pair_usage / 20.0);
    
    // 特征6: 平面负载均衡度
    vector<int> plane_load(topo.P, 0);
    for (int m = 0; m < topo.M; ++m) {
        int plane = m / topo.OXCsPerPlane;
        for (int p = 0; p < topo.R; ++p) {
            if (oxc_conn[m][p] != -1) plane_load[plane]++;
        }
    }
    double avg_plane_load = accumulate(plane_load.begin(), plane_load.end(), 0.0) / topo.P;
    double load_variance = 0.0;
    for (int p = 0; p < topo.P; ++p) {
        load_variance += pow(plane_load[p] - avg_plane_load, 2);
    }
    features[5] = sqrt(load_variance / topo.P) / (avg_plane_load + 1);
    
    return features;
}

// ==================== 动作特征提取器（保持与原代码兼容）====================
vector<double> extract_action_features(
    const Topology& topo,
    const array<int, 5>& cand,
    int gA, int gB,
    const vector<vector<int>>& spine_load,
    const vector<vector<int>>& group_plane_usage,
    const vector<vector<int>>& oxc_conn) {
    
    vector<double> features(RL_ACTION_DIM, 0.0);
    int sA = cand[0], oxc = cand[2], sB = cand[3];
    int plane = oxc / topo.OXCsPerPlane;
    
    // 特征1: 路径负载（标准化）
    features[0] = (spine_load[sA][oxc] + spine_load[sB][oxc]) / (2.0 * MAX_LOAD_THRESHOLD);
    
    // 特征2: 平面使用历史
    int hist_usage = group_plane_usage[gA][plane] + group_plane_usage[gB][plane];
    features[1] = min(1.0, hist_usage / 10.0);
    
    // 特征3: OXC端口占用率
    int used_ports = 0;
    for (int p = 0; p < topo.R; ++p) {
        if (oxc_conn[oxc][p] != -1) used_ports++;
    }
    features[2] = static_cast<double>(used_ports) / topo.R;
    
    // 特征4: 调整成本估计
    int pA = topo.get_port(oxc, gA, sA, cand[1]);
    int pB = topo.get_port(oxc, gB, sB, cand[4]);
    int cost = (oxc_conn[oxc][pA] == -1 && oxc_conn[oxc][pB] == -1) ? 0 : 1;
    features[3] = cost;
    
    // 特征5: Spine负载均衡度
    double load_diff = abs(spine_load[sA][oxc] - spine_load[sB][oxc]);
    features[4] = load_diff / (MAX_LOAD_THRESHOLD + 1e-10);
    
    return features;
}

// ==================== 轻量级神经网络（仅使用标准库）====================
class LightweightNeuralNetwork {
private:
    struct Layer {
        vector<vector<double>> weights;  // [output_dim][input_dim]
        vector<double> biases;           // [output_dim]
        vector<double> activation;       // 激活值
        vector<double> pre_activation;   // 线性输出（用于反向传播）
        
        Layer(int input_dim, int output_dim) {
            // Xavier初始化
            double scale = sqrt(2.0 / (input_dim + output_dim));
            uniform_real_distribution<double> dist(-scale, scale);
            
            weights.resize(output_dim, vector<double>(input_dim));
            biases.resize(output_dim, 0.0);
            activation.resize(output_dim, 0.0);
            pre_activation.resize(output_dim, 0.0);
            
            for (int i = 0; i < output_dim; ++i) {
                for (int j = 0; j < input_dim; ++j) {
                    weights[i][j] = dist(rng);
                }
            }
        }
        
        // ReLU激活函数（带泄漏）
        static double relu(double x) { return max(0.1 * x, x); }  // Leaky ReLU
        static double relu_derivative(double x) { return x > 0 ? 1.0 : 0.1; }
        
        // 前向传播
        void forward(const vector<double>& input) {
            int output_dim = weights.size();
            int input_dim = weights[0].size();
            
            for (int i = 0; i < output_dim; ++i) {
                pre_activation[i] = biases[i];
                for (int j = 0; j < input_dim; ++j) {
                    pre_activation[i] += weights[i][j] * input[j];
                }
                activation[i] = relu(pre_activation[i]);
            }
        }
        
        // 反向传播
        void backward(const vector<double>& input,
                     const vector<double>& delta_next,
                     vector<double>& delta_current,
                     vector<vector<double>>& weight_grad,
                     vector<double>& bias_grad,
                     double learning_rate) {
            int output_dim = weights.size();
            int input_dim = weights[0].size();
            
            for (int i = 0; i < output_dim; ++i) {
                double grad = delta_next[i] * relu_derivative(pre_activation[i]);
                
                // 更新权重梯度
                for (int j = 0; j < input_dim; ++j) {
                    weight_grad[i][j] += grad * input[j];
                }
                bias_grad[i] += grad;
                
                // 传播梯度到前一层
                if (delta_current.size() == input_dim) {
                    for (int j = 0; j < input_dim; ++j) {
                        delta_current[j] += grad * weights[i][j];
                    }
                }
            }
        }
        
        // 更新权重
        void update(const vector<vector<double>>& weight_grad,
                   const vector<double>& bias_grad,
                   double learning_rate,
                   int batch_size) {
            int output_dim = weights.size();
            int input_dim = weights[0].size();
            
            for (int i = 0; i < output_dim; ++i) {
                for (int j = 0; j < input_dim; ++j) {
                    weights[i][j] -= learning_rate * weight_grad[i][j] / batch_size;
                }
                biases[i] -= learning_rate * bias_grad[i] / batch_size;
            }
        }
    };
    
    vector<Layer> layers;
    int input_dim;
    int output_dim;
    
public:
    LightweightNeuralNetwork(int input_dim, int output_dim) 
        : input_dim(input_dim), output_dim(output_dim) {
        
        // 构建网络：输入层 -> 隐藏层 -> 输出层
        if (RL_HIDDEN_DIM > 0) {
            layers.emplace_back(input_dim, RL_HIDDEN_DIM);
            layers.emplace_back(RL_HIDDEN_DIM, output_dim);
        } else {
            layers.emplace_back(input_dim, output_dim);
        }
    }
    
    // 前向传播
    vector<double> forward(const vector<double>& input) {
        vector<double> current = input;
        
        for (size_t l = 0; l < layers.size(); ++l) {
            layers[l].forward(current);
            current = layers[l].activation;
        }
        
        // 输出层使用softmax（如果是概率输出）
        if (output_dim > 1) {
            return softmax(current);
        }
        return current;  // 单输出（Q值）
    }
    
    // 计算softmax
    vector<double> softmax(const vector<double>& logits) {
        vector<double> result(logits.size());
        double max_val = *max_element(logits.begin(), logits.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            result[i] = exp(logits[i] - max_val);
            sum += result[i];
        }
        
        if (sum > 0) {
            for (auto& val : result) {
                val /= sum;
            }
        }
        
        return result;
    }
    
    // 训练一个batch
    void train_batch(const vector<vector<double>>& inputs,
                    const vector<vector<double>>& targets,
                    double learning_rate) {
        
        int batch_size = inputs.size();
        if (batch_size == 0) return;
        
        // 存储每层的梯度
        vector<vector<vector<double>>> weight_grads;
        vector<vector<double>> bias_grads;
        
        for (auto& layer : layers) {
            weight_grads.push_back(vector<vector<double>>(
                layer.weights.size(), 
                vector<double>(layer.weights[0].size(), 0.0)
            ));
            bias_grads.push_back(vector<double>(layer.biases.size(), 0.0));
        }
        
        // 前向和反向传播
        for (int b = 0; b < batch_size; ++b) {
            // 前向传播
            forward(inputs[b]);
            
            // 计算输出层梯度（均方误差损失）
            int output_dim = layers.back().activation.size();
            vector<double> delta(output_dim);
            const vector<double>& output = layers.back().activation;
            
            for (int i = 0; i < output_dim; ++i) {
                delta[i] = output[i] - targets[b][i];
            }
            
            // 反向传播
            vector<double> delta_current(layers.back().weights[0].size(), 0.0);
            
            for (int l = layers.size() - 1; l >= 0; --l) {
                vector<double> delta_next = (l == layers.size() - 1) ? delta : delta_current;
                vector<double> new_delta_current(layers[l].weights[0].size(), 0.0);
                
                const vector<double>& layer_input = (l == 0) ? inputs[b] : layers[l-1].activation;
                
                layers[l].backward(layer_input, delta_next, new_delta_current,
                                  weight_grads[l], bias_grads[l], learning_rate);
                
                delta_current = new_delta_current;
            }
        }
        
        // 更新权重
        for (size_t l = 0; l < layers.size(); ++l) {
            layers[l].update(weight_grads[l], bias_grads[l], learning_rate, batch_size);
        }
    }
    
    // 获取网络参数数量
    int get_num_params() const {
        int total = 0;
        for (const auto& layer : layers) {
            total += layer.weights.size() * layer.weights[0].size() + layer.biases.size();
        }
        return total;
    }
};

// ==================== 策略微调器（论文方法简化版）====================
class PolicyFineTuner {
private:
    LightweightNeuralNetwork policy_net;  // 策略网络
    LightweightNeuralNetwork value_net;   // 价值网络
    double learning_rate;
    double gamma;
    int horizon;
    
    // 经验回放缓冲区
    struct Experience {
        vector<double> state;
        vector<double> action_features;
        double reward;
        vector<double> next_state;
        bool done;
    };
    deque<Experience> replay_buffer;
    int max_buffer_size;
    
public:
    PolicyFineTuner(int state_dim, int action_dim, 
                   double lr = RL_LEARNING_RATE, double g = RL_DISCOUNT, 
                   int h = RL_PLANNING_HORIZON, int buffer_size = RL_MAX_BUFFER_SIZE)
        : policy_net(state_dim + action_dim, 1)  // Q网络：状态+动作 -> Q值
        , value_net(state_dim, 1)                // V网络：状态 -> 状态价值
        , learning_rate(lr)
        , gamma(g)
        , horizon(h)
        , max_buffer_size(buffer_size) {}
    
    // 计算动作的Q值
    double compute_q_value(const vector<double>& state, const vector<double>& action_features) {
        vector<double> combined(state);
        combined.insert(combined.end(), action_features.begin(), action_features.end());
        vector<double> result = policy_net.forward(combined);
        return result[0];
    }
    
    // 计算状态价值
    double compute_state_value(const vector<double>& state) {
        vector<double> result = value_net.forward(state);
        return result[0];
    }
    
    // 选择动作：基于当前策略选择最佳动作
    int select_best_action(const vector<double>& state,
                          const vector<vector<double>>& action_features_list) {
        
        if (action_features_list.empty()) return -1;
        
        int best_action = 0;
        double best_q = -1e18;
        
        // 计算每个动作的Q值
        for (size_t i = 0; i < action_features_list.size(); ++i) {
            double q = compute_q_value(state, action_features_list[i]);
            if (q > best_q) {
                best_q = q;
                best_action = i;
            }
        }
        
        // ε-greedy探索
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < RL_EPSILON) {
            return rng() % action_features_list.size();
        }
        
        return best_action;
    }
    
    // 添加经验到缓冲区
    void add_experience(const vector<double>& state,
                       const vector<double>& action_features,
                       double reward,
                       const vector<double>& next_state,
                       bool done) {
        
        replay_buffer.push_back({state, action_features, reward, next_state, done});
        
        if (replay_buffer.size() > max_buffer_size) {
            replay_buffer.pop_front();
        }
    }
    
    // 从缓冲区采样
    vector<Experience> sample_experiences(int batch_size) {
        if ((int)replay_buffer.size() < batch_size) {
            return vector<Experience>(replay_buffer.begin(), replay_buffer.end());
        }
        
        vector<Experience> batch;
        vector<int> indices(replay_buffer.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);
        
        for (int i = 0; i < batch_size; ++i) {
            batch.push_back(replay_buffer[indices[i]]);
        }
        
        return batch;
    }
    
    // 快速微调（在线学习）
    void fast_fine_tune(int num_updates = 10) {
        if (replay_buffer.size() < RL_BATCH_SIZE) return;
        
        for (int update = 0; update < num_updates; ++update) {
            // 采样经验
            auto batch = sample_experiences(RL_BATCH_SIZE);
            
            // 准备训练数据
            vector<vector<double>> policy_inputs;
            vector<vector<double>> policy_targets;
            vector<vector<double>> value_inputs;
            vector<vector<double>> value_targets;
            
            for (const auto& exp : batch) {
                // 计算TD目标
                double target_value = exp.reward;
                if (!exp.done) {
                    double next_value = compute_state_value(exp.next_state);
                    target_value += gamma * next_value;
                }
                
                // 价值网络训练
                value_inputs.push_back(exp.state);
                value_targets.push_back({target_value});
                
                // 策略网络训练（Q-learning更新）
                policy_inputs.push_back(exp.state);
                policy_inputs.back().insert(policy_inputs.back().end(), 
                                          exp.action_features.begin(), 
                                          exp.action_features.end());
                
                // 当前Q值
                double current_q = compute_q_value(exp.state, exp.action_features);
                // Q-learning目标：r + γ * max_a' Q(s', a')
                policy_targets.push_back({target_value});
            }
            
            // 更新网络
            if (!value_inputs.empty()) {
                value_net.train_batch(value_inputs, value_targets, learning_rate);
            }
            if (!policy_inputs.empty()) {
                policy_net.train_batch(policy_inputs, policy_targets, learning_rate);
            }
        }
    }
    
    // 多步规划（模拟rollout）
    double plan_with_rollout(const vector<double>& start_state,
                            const vector<double>& action_features,
                            int num_rollouts = 15) {
        
        double total_return = 0.0;
        
        for (int r = 0; r < num_rollouts; ++r) {
            vector<double> current_state = start_state;
            vector<double> current_action = action_features;
            double rollout_return = 0.0;
            
            for (int step = 0; step < horizon; ++step) {
                // 估计即时奖励（基于动作特征）
                double reward = estimate_reward(current_state, current_action);
                rollout_return += pow(gamma, step) * reward;
                
                // 模拟状态转移（简化）
                current_state = simulate_state_transition(current_state, current_action);
                
                // 选择下一个动作（基于当前策略）
                // 这里需要候选动作列表，但我们简化处理
                if (step < horizon - 1) {
                    // 随机生成一个动作特征
                    vector<double> next_action(RL_ACTION_DIM);
                    for (int i = 0; i < RL_ACTION_DIM; ++i) {
                        uniform_real_distribution<double> dist(0.0, 1.0);
                        next_action[i] = dist(rng);
                    }
                    current_action = next_action;
                }
            }
            
            // 添加终值估计
            double terminal_value = compute_state_value(current_state);
            rollout_return += pow(gamma, horizon) * terminal_value;
            
            total_return += rollout_return;
        }
        
        return total_return / num_rollouts;
    }
    
private:
    // 估计奖励（简化）
    double estimate_reward(const vector<double>& state, const vector<double>& action_features) {
        // 奖励设计：负载均衡 + 低冲突 + 高成功率
        double reward = 0.0;
        
        // 基于状态特征的奖励
        if (state.size() > 1) {
            // 低负载奖励
            reward += 1.0 - state[1];  // 特征1是负载
            // 均衡奖励
            reward += 1.0 - state[5];  // 特征6是负载不均衡度
        }
        
        // 基于动作特征的奖励
        if (action_features.size() > 0) {
            // 低负载路径奖励
            reward += 1.0 - action_features[0];  // 特征0是路径负载
            // 低成本奖励
            reward += 1.0 - action_features[3];  // 特征3是调整成本
            // 均衡奖励
            reward += 1.0 - action_features[4];  // 特征4是负载差异
        }
        
        return reward / 5.0;  // 归一化
    }
    
    // 模拟状态转移（简化）
    vector<double> simulate_state_transition(const vector<double>& state,
                                           const vector<double>& action_features) {
        // 简化：在状态上添加小的随机噪声
        vector<double> next_state = state;
        uniform_real_distribution<double> noise_dist(-0.05, 0.05);
        
        for (auto& val : next_state) {
            val = clamp(val + noise_dist(rng), 0.0, 1.0);
        }
        
        return next_state;
    }
};

// ==================== 改进的组优化器（集成策略微调）====================
class ImprovedGroupOptimizer {
    struct PendingFlow {
        size_t flow_idx;
        int gA, lA, gB, lB;
    };
    
    const Topology& topo;
    vector<vector<int>>& group_plane_usage;
    const vector<vector<double>>& mcf_guide_weights;
    PolicyFineTuner policy_tuner;
    
public:
    ImprovedGroupOptimizer(const Topology& t, vector<vector<int>>& usage, 
                          const vector<vector<double>>& mcf_weights)
        : topo(t), group_plane_usage(usage), mcf_guide_weights(mcf_weights),
          policy_tuner(RL_STATE_DIM, RL_ACTION_DIM) {}
    
    vector<array<int, 5>> optimize(
        const vector<size_t>& idx,
        const vector<tuple<int, int, int, int>>& flows,
        const vector<vector<int>>& init_conn,
        const ConflictAnalyzer& ca,
        double& bestScore,
        vector<vector<int>>& finalConn) {
        
        bestScore = -1e18;
        finalConn = init_conn;

        vector<array<int, 5>> routes(flows.size(), {-1, -1, -1, -1, -1});
        vector<vector<int>> conn = init_conn;
        
        // 第一阶段：尝试分配所有流，将失败的收集到待规划池
        vector<PendingFlow> pending_flows;
        
        for (size_t i = 0; i < idx.size(); ++i) {
            size_t flow_idx = idx[i];
            auto [gA, lA, gB, lB] = flows[flow_idx];
            
            // 尝试随机搜索（基于MCF引导）
            const vector<double>& current_mcf_weight = 
                (flow_idx < mcf_guide_weights.size()) ? mcf_guide_weights[flow_idx] : vector<double>();
            
            bool found = false;
            for (int att = 0; att < 2000 && !found; ++att) {
                int pl;
                if (!current_mcf_weight.empty()) {
                    vector<double> cumsum(topo.P);
                    partial_sum(current_mcf_weight.begin(), current_mcf_weight.end(), cumsum.begin());
                    double total_w = cumsum.back();
                    uniform_real_distribution<double> dist(0.0, total_w);
                    double r = dist(rng);
                    pl = lower_bound(cumsum.begin(), cumsum.end(), r) - cumsum.begin();
                    pl = min(pl, topo.P - 1);
                } else {
                    pl = rng() % topo.P;
                }

                int sA = pl * topo.SpinesPerPlane + (rng() % topo.SpinesPerPlane);
                int sB = pl * topo.SpinesPerPlane + (rng() % topo.SpinesPerPlane);
                int oxc = pl * topo.OXCsPerPlane + (rng() % topo.OXCsPerPlane);
                int kA = rng() % topo.K;
                int kB = rng() % topo.K;

                if (establish_connection(topo, conn, gA, lA, gB, lB, sA, kA, oxc, sB, kB, 
                                        group_plane_usage, routes[flow_idx])) {
                    found = true;
                }
            }
            
            // 如果随机搜索失败，尝试有限穷举搜索
            if (!found) {
                for (int pl = 0; pl < min(2, topo.P) && !found; ++pl) {
                    for (int sA = pl * topo.SpinesPerPlane; sA < (pl + 1) * topo.SpinesPerPlane && !found; ++sA) {
                        for (int oidx = 0; oidx < topo.OXCsPerPlane && !found; ++oidx) {
                            int oxc = pl * topo.OXCsPerPlane + oidx;
                            for (int sB = pl * topo.SpinesPerPlane; sB < (pl + 1) * topo.SpinesPerPlane && !found; ++sB) {
                                for (int kA = 0; kA < topo.K && !found; ++kA) {
                                    for (int kB = 0; kB < topo.K && !found; ++kB) {
                                        if (establish_connection(topo, conn, gA, lA, gB, lB, sA, kA, oxc, sB, kB,
                                                                group_plane_usage, routes[flow_idx])) {
                                            found = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // 如果还是没找到，加入到待规划池
            if (!found) {
                pending_flows.push_back({flow_idx, gA, lA, gB, lB});
            }
        }
        
        // 第二阶段：处理待规划池中的流，使用策略微调进行协同规划
        if (!pending_flows.empty()) {
            auto current_spine_load = compute_spine_load(topo, conn);
            
            // 为每个待规划流进行规划
            for (int flow_idx = 0; flow_idx < (int)pending_flows.size(); ++flow_idx) {
                const auto& pending = pending_flows[flow_idx];
                size_t f_idx = pending.flow_idx;
                int gA = pending.gA, lA = pending.lA, gB = pending.gB, lB = pending.lB;
                
                // 生成所有候选路径
                vector<array<int, 5>> candidates;
                vector<vector<double>> action_features_list;
                
                for (int pl = 0; pl < topo.P; ++pl) {
                    for (int sA = pl * topo.SpinesPerPlane; sA < (pl + 1) * topo.SpinesPerPlane; ++sA) {
                        for (int oxc = pl * topo.OXCsPerPlane; oxc < (pl + 1) * topo.OXCsPerPlane; ++oxc) {
                            for (int sB = pl * topo.SpinesPerPlane; sB < (pl + 1) * topo.SpinesPerPlane; ++sB) {
                                for (int kA = 0; kA < topo.K; ++kA) {
                                    for (int kB = 0; kB < topo.K; ++kB) {
                                        if (is_path_valid(topo, conn, gA, lA, gB, lB, sA, kA, oxc, sB, kB)) {
                                            array<int, 5> cand = {sA, kA, oxc, sB, kB};
                                            candidates.push_back(cand);
                                            
                                            // 提取动作特征
                                            auto action_features = extract_action_features(
                                                topo, cand, gA, gB, current_spine_load, 
                                                group_plane_usage, conn);
                                            action_features_list.push_back(action_features);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (!candidates.empty()) {
                    // 提取当前状态特征
                    int remaining_flows = pending_flows.size() - flow_idx;
                    auto state_features = extract_state_features(
                        topo, gA, gB, current_spine_load, conn, 
                        group_plane_usage, remaining_flows);
                    
                    // 使用策略微调选择动作
                    int chosen_idx = policy_tuner.select_best_action(state_features, action_features_list);
                    
                    if (chosen_idx >= 0 && chosen_idx < (int)candidates.size()) {
                        auto cand = candidates[chosen_idx];
                        
                        // 尝试建立连接
                        bool success = establish_connection(topo, conn, gA, lA, gB, lB, 
                                                           cand[0], cand[1], cand[2], cand[3], cand[4],
                                                           group_plane_usage, routes[f_idx]);
                        
                        // 计算奖励
                        double reward = success ? 1.0 : -1.0;
                        if (success) {
                            // 成功建立连接，额外奖励基于负载均衡
                            auto action_features = action_features_list[chosen_idx];
                            reward += 1.0 - action_features[0];  // 低负载奖励
                            reward += 1.0 - action_features[4];  // 均衡奖励
                        }
                        
                        // 获取下一个状态
                        auto next_spine_load = compute_spine_load(topo, conn);
                        auto next_state_features = extract_state_features(
                            topo, gA, gB, next_spine_load, conn, 
                            group_plane_usage, max(0, remaining_flows - 1));
                        
                        // 添加经验到缓冲区
                        policy_tuner.add_experience(state_features, 
                                                   action_features_list[chosen_idx],
                                                   reward, next_state_features, false);
                        
                        // 如果成功，继续处理下一个流
                        if (success) {
                            // 每处理几个流后进行一次微调
                            if (flow_idx % 5 == 4) {
                                policy_tuner.fast_fine_tune(3);
                            }
                            continue;
                        }
                    }
                }
                
                // 如果策略微调没有找到路径或建立连接失败，使用原策略的智能兜底
                bool assigned = false;
                
                // 智能兜底：寻找所有可能路径，选择负载最轻的
                for (int pl = 0; pl < topo.P && !assigned; ++pl) {
                    for (int sA = pl * topo.SpinesPerPlane; sA < (pl + 1) * topo.SpinesPerPlane && !assigned; ++sA) {
                        for (int oxc = pl * topo.OXCsPerPlane; oxc < (pl + 1) * topo.OXCsPerPlane && !assigned; ++oxc) {
                            for (int sB = pl * topo.SpinesPerPlane; sB < (pl + 1) * topo.SpinesPerPlane && !assigned; ++sB) {
                                for (int kA = 0; kA < topo.K && !assigned; ++kA) {
                                    for (int kB = 0; kB < topo.K && !assigned; ++kB) {
                                        if (establish_connection(topo, conn, gA, lA, gB, lB, sA, kA, oxc, sB, kB,
                                                                group_plane_usage, routes[f_idx])) {
                                            assigned = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 终极兜底：强制建立连接
                if (!assigned) {
                    for (int oxc = 0; oxc < topo.M && !assigned; ++oxc) {
                        int plane = oxc / topo.OXCsPerPlane;
                        for (int sA_local = 0; sA_local < topo.SpinesPerPlane && !assigned; ++sA_local) {
                            int sA = plane * topo.SpinesPerPlane + sA_local;
                            for (int sB_local = 0; sB_local < topo.SpinesPerPlane && !assigned; ++sB_local) {
                                int sB = plane * topo.SpinesPerPlane + sB_local;
                                for (int kA = 0; kA < topo.K && !assigned; ++kA) {
                                    for (int kB = 0; kB < topo.K && !assigned; ++kB) {
                                        int pA = topo.get_port(oxc, gA, sA, kA);
                                        int pB = topo.get_port(oxc, gB, sB, kB);
                                        
                                        if (pA >= 0 && pA < topo.R && pB >= 0 && pB < topo.R && pA != pB) {
                                            // 清除原有连接
                                            if (conn[oxc][pA] != -1) {
                                                int old_p = conn[oxc][pA];
                                                conn[oxc][old_p] = -1;
                                            }
                                            if (conn[oxc][pB] != -1) {
                                                int old_p = conn[oxc][pB];
                                                conn[oxc][old_p] = -1;
                                            }
                                            
                                            // 建立新连接
                                            conn[oxc][pA] = pB;
                                            conn[oxc][pB] = pA;
                                            
                                            routes[f_idx] = {sA, kA, oxc, sB, kB};
                                            int plane_id = oxc / topo.OXCsPerPlane;
                                            group_plane_usage[gA][plane_id]++;
                                            group_plane_usage[gB][plane_id]++;
                                            
                                            assigned = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // 在处理完所有待规划流后进行一次完整的微调
            policy_tuner.fast_fine_tune(10);
        }
        
        // 最终验证：确保组内所有流都有有效的路由和连接
        for (size_t i = 0; i < idx.size(); ++i) {
            size_t flow_idx = idx[i];
            auto& route = routes[flow_idx];
            
            if (route[0] == -1 || route[2] == -1 || route[3] == -1) {
                int gA, lA, gB, lB;
                tie(gA, lA, gB, lB) = flows[flow_idx];
                
                for (int oxc = 0; oxc < topo.M; ++oxc) {
                    int plane = oxc / topo.OXCsPerPlane;
                    for (int sA_local = 0; sA_local < topo.SpinesPerPlane; ++sA_local) {
                        int sA = plane * topo.SpinesPerPlane + sA_local;
                        for (int sB_local = 0; sB_local < topo.SpinesPerPlane; ++sB_local) {
                            int sB = plane * topo.SpinesPerPlane + sB_local;
                            for (int kA = 0; kA < topo.K; ++kA) {
                                for (int kB = 0; kB < topo.K; ++kB) {
                                    int pA = topo.get_port(oxc, gA, sA, kA);
                                    int pB = topo.get_port(oxc, gB, sB, kB);
                                    
                                    if (pA >= 0 && pA < topo.R && pB >= 0 && pB < topo.R && pA != pB) {
                                        if (conn[oxc][pA] != -1) {
                                            int old_p = conn[oxc][pA];
                                            conn[oxc][old_p] = -1;
                                        }
                                        if (conn[oxc][pB] != -1) {
                                            int old_p = conn[oxc][pB];
                                            conn[oxc][old_p] = -1;
                                        }
                                        
                                        conn[oxc][pA] = pB;
                                        conn[oxc][pB] = pA;
                                        
                                        route = {sA, kA, oxc, sB, kB};
                                        goto route_fixed;
                                    }
                                }
                            }
                        }
                    }
                }
                route_fixed:;
            }
            
            // 验证连接是否正确
            if (route[0] != -1 && route[2] != -1 && route[3] != -1) {
                int oxc = route[2];
                int pA = topo.get_port(oxc, get<0>(flows[flow_idx]), route[0], route[1]);
                int pB = topo.get_port(oxc, get<2>(flows[flow_idx]), route[3], route[4]);
                
                if (conn[oxc][pA] != pB || conn[oxc][pB] != pA) {
                    if (conn[oxc][pA] != -1) {
                        int old_p = conn[oxc][pA];
                        conn[oxc][old_p] = -1;
                    }
                    if (conn[oxc][pB] != -1) {
                        int old_p = conn[oxc][pB];
                        conn[oxc][old_p] = -1;
                    }
                    
                    conn[oxc][pA] = pB;
                    conn[oxc][pB] = pA;
                }
            }
        }
        
        finalConn = move(conn);
        
        // 提取当前组的路由
        vector<array<int, 5>> group_routes;
        for (size_t i = 0; i < idx.size(); ++i) {
            group_routes.push_back(routes[idx[i]]);
        }
        
        return group_routes;
    }
};

// ==================== 批量优化主控 ====================
class BatchOptimizer {
    const Topology& topo;
    vector<vector<int>> finalConn;
    vector<vector<int>> group_plane_usage;

public:
    BatchOptimizer(const Topology& t) : topo(t) {
        group_plane_usage.assign(topo.N, vector<int>(topo.P, 0));
    }

    vector<array<int, 5>> optimize(
        const vector<tuple<int, int, int, int>>& flows,
        const vector<vector<int>>& init) {
        
        LightweightMCFGuide mcf_guide(topo, flows);
        vector<vector<double>> mcf_weights = mcf_guide.run();

        // 【关键修改】使用改进版ConflictAnalyzer，传入topo引用
        ConflictAnalyzer ca(flows, topo.convergence_ratio, topo);
        auto groups = ca.group(CONFLICT_THRESHOLD);

        vector<array<int, 5>> all(flows.size());
        vector<vector<int>> conn = init;
        
        for (const auto& g : groups) {
            double sc;
            vector<vector<int>> newConn;
            ImprovedGroupOptimizer go(topo, group_plane_usage, mcf_weights);
            auto routes = go.optimize(g, flows, conn, ca, sc, newConn);
            conn = move(newConn);
            for (size_t i = 0; i < g.size(); ++i) {
                all[g[i]] = routes[i];
            }
        }
        
        // 最终全局验证
        for (size_t i = 0; i < flows.size(); ++i) {
            auto& route = all[i];
            auto [gA, lA, gB, lB] = flows[i];
            
            if (route[0] == -1 || route[2] == -1 || route[3] == -1) {
                for (int oxc = 0; oxc < topo.M; ++oxc) {
                    int plane = oxc / topo.OXCsPerPlane;
                    for (int sA_local = 0; sA_local < topo.SpinesPerPlane; ++sA_local) {
                        int sA = plane * topo.SpinesPerPlane + sA_local;
                        for (int sB_local = 0; sB_local < topo.SpinesPerPlane; ++sB_local) {
                            int sB = plane * topo.SpinesPerPlane + sB_local;
                            for (int kA = 0; kA < topo.K; ++kA) {
                                for (int kB = 0; kB < topo.K; ++kB) {
                                    int pA = topo.get_port(oxc, gA, sA, kA);
                                    int pB = topo.get_port(oxc, gB, sB, kB);
                                    
                                    if (pA >= 0 && pA < topo.R && pB >= 0 && pB < topo.R && pA != pB) {
                                        if (conn[oxc][pA] != -1) {
                                            int old_p = conn[oxc][pA];
                                            conn[oxc][old_p] = -1;
                                        }
                                        if (conn[oxc][pB] != -1) {
                                            int old_p = conn[oxc][pB];
                                            conn[oxc][old_p] = -1;
                                        }
                                        
                                        conn[oxc][pA] = pB;
                                        conn[oxc][pB] = pA;
                                        
                                        route = {sA, kA, oxc, sB, kB};
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                int oxc = route[2];
                int pA = topo.get_port(oxc, gA, route[0], route[1]);
                int pB = topo.get_port(oxc, gB, route[3], route[4]);
                
                if (conn[oxc][pA] != pB || conn[oxc][pB] != pA) {
                    if (conn[oxc][pA] != -1) {
                        int old_p = conn[oxc][pA];
                        conn[oxc][old_p] = -1;
                    }
                    if (conn[oxc][pB] != -1) {
                        int old_p = conn[oxc][pB];
                        conn[oxc][old_p] = -1;
                    }
                    
                    conn[oxc][pA] = pB;
                    conn[oxc][pB] = pA;
                }
            }
        }
        
        finalConn = move(conn);
        return all;
    }

    const vector<vector<int>>& getFinalConn() const {
        return finalConn;
    }
};

// ==================== 主函数 ====================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, S, L, M, K, P;
    cin >> N >> S >> L >> M >> K >> P;
    Topology topo(N, S, L, M, K, P);

    vector<vector<int>> gConn(M, vector<int>(topo.R, -1));

    for (int q = 0; q < 5; ++q) {
        int Q;
        cin >> Q;
        vector<tuple<int, int, int, int>> flows(Q);
        for (int i = 0; i < Q; ++i) {
            int gA, lA, gB, lB;
            cin >> gA >> lA >> gB >> lB;
            flows[i] = {gA, lA, gB, lB};
        }

        BatchOptimizer opt(topo);
        auto routes = opt.optimize(flows, gConn);
        gConn = opt.getFinalConn();

        // 输出OXC连接状态
        for (int m = 0; m < M; ++m) {
            for (int p = 0; p < topo.R; ++p) {
                if (p) cout << ' ';
                cout << gConn[m][p];
            }
            cout << '\n';
        }

        // 输出路由
        for (auto& r : routes) {
            cout << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << r[3] << ' ' << r[4] << '\n';
        }
    }
    return 0;
}