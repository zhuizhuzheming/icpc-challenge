#include <bits/stdc++.h>
using namespace std;

// ==================== 常量配置 ====================
const int SMALL_N = 30;
const int K_NEIGHBORS = 3;
const double EPSILON_ENTROPY = 0.5;
const double SURPLUS_UNFINISHED = -5.0;
const double SURPLUS_FINISHED = 2.0;
const double POWER_PENALTY_COEF = 0.05;
const double SOCIAL_BONUS = 0.3;
const double SOCIAL_BID_WEIGHT = 0.2;

const int GRACE_PERIOD = 2;
const double AGING_WEIGHT = 0.15;
const double FEASIBILITY_MARGIN = 0.8;
const double BASE_EXP_COEF = 2.2;
const double W = 192.0;
const double EPS = 1e-9;
const double FAIRNESS_WEIGHT = 0.3;

const int NUM_ACTIONS = 10;
const int STATE_DIM = 6;
const double ALPHA = 0.05;
const double GAMMA = 0.995;
const double LAMBDA = 0.9;
const double EPSILON_START = 1.0;
const double EPSILON_END = 0.1;

const double INITIAL_TEMP = 100.0;
const double COOLING_RATE = 0.995;
const int SA_ITERATIONS = 100;
const double MIN_TEMP = 0.01;

// ==================== Global Variables ====================
int N, K, T, RBG_COUNT;
vector<vector<vector<vector<double>>>> s0;
vector<vector<vector<vector<double>>>> d;

struct Frame {
    int user, t0, t1;
    double TBS;
    inline bool is_active_at(int t) const { return t0 <= t && t <= t1; }
};
vector<Frame> frames;

vector<vector<vector<vector<double>>>> p;
vector<vector<vector<double>>> rbg_power;

struct FrameState {
    double accumulated_bits = 0.0;
    double total_power_used = 0.0;
    bool finished = false;
    bool updated_beta = false;
};
vector<FrameState> frame_state;
vector<vector<int>> active_frames_per_tti;
vector<int> last_scheduled;

vector<vector<vector<double>>> cross_cell_power;

// ==================== Forward Declarations ====================
double simulated_annealing_optimize(int r, int k, int t, int selected_user, double rem_bits, double budget);
pair<double, double> optimize_joint_power_grid(int r, int k, int t, int u1, int u2, double rem1, double rem2, double budget);

// ==================== Helper Functions ====================
void update_cross_cell_power(int r, int t) {
    for (int k = 0; k < K; ++k) {
        double total = 0.0;
        for (int kp = 0; kp < K; ++kp) {
            if (kp == k) continue;
            for (int np = 0; np < N; ++np) {
                if (p[r][kp][t][np] > EPS) {
                    total += s0[r][kp][t][np] * p[r][kp][t][np];
                }
            }
        }
        cross_cell_power[r][k][t] = total;
    }
}

double compute_actual_rate(int n, int k, int t, int r) {
    if (p[r][k][t][n] <= EPS) return 0.0;
    double numerator = s0[r][k][t][n] * p[r][k][t][n];
    for (int m = 0; m < N; ++m) {
        if (m != n && p[r][k][t][m] > EPS) {
            numerator *= exp(d[m][r][k][n]);
        }
    }
    double inter = cross_cell_power[r][k][t];
    double sinr = numerator / (1.0 + inter);
    if (sinr <= EPS) return 0.0;
    return W * log2(1.0 + sinr);
}

double calculate_urgency(int j, int t) {
    const Frame& f = frames[j];
    int rem_ttis = f.t1 - t + 1;
    if (rem_ttis <= 0) return -1e9;
    double rem_bits = f.TBS - frame_state[j].accumulated_bits;
    if (rem_bits <= EPS) return -1e9;
    double progress = 1.0 - (double)(f.t1 - t + 1) / max(1, f.t1 - f.t0 + 1);
    double frame_urgency = exp(BASE_EXP_COEF * progress) * log(rem_bits + 1.0);
    int since_last = t - last_scheduled[f.user];
    int starvation = max(0, since_last - GRACE_PERIOD);
    double fairness_bonus = FAIRNESS_WEIGHT * starvation;
    return frame_urgency + fairness_bonus;
}

bool is_feasible(int j, int t, int r, int k) {
    const Frame& f = frames[j];
    int rem_ttis = f.t1 - t + 1;
    double rem_bits = f.TBS - frame_state[j].accumulated_bits;
    if (rem_bits <= EPS) return true;
    double max_sinr = s0[r][k][t][f.user] * 4.0 / (1.0 + cross_cell_power[r][k][t]);
    double max_rate_per_tti = W * log2(1.0 + max_sinr);
    return max_rate_per_tti * rem_ttis >= rem_bits * FEASIBILITY_MARGIN;
}

// ==================== Bidding Agent ====================
struct BiddingAgent {
    vector<pair<double, double>> particles;
    vector<double> weights;
    mt19937 rng;

    BiddingAgent() {
        auto now = chrono::steady_clock::now().time_since_epoch().count();
        rng.seed(static_cast<unsigned int>(now & 0xFFFFFFFF));
        uniform_real_distribution<double> d1(0.5, 2.0);
        uniform_real_distribution<double> d2(0.1, 3.0);
        particles.resize(SMALL_N);
        weights.assign(SMALL_N, 1.0 / SMALL_N);
        for (int i = 0; i < SMALL_N; ++i) {
            particles[i] = {d1(rng), d2(rng)};
        }
    }

    double compute_bid(double urgency, int& sampled_idx) {
        discrete_distribution<int> dd(weights.begin(), weights.end());
        sampled_idx = dd(rng);
        auto [θ1, θ2] = particles[sampled_idx];
        if (θ2 <= 1e-6) return urgency;
        return log(1.0 + θ1 * θ2 * urgency) / θ2;
    }

    double compute_social_bid(double self_urgency, double max_other_urgency, int& sampled_idx) {
        double self_bid = compute_bid(self_urgency, sampled_idx);
        if (max_other_urgency <= 0) return self_bid;
        double other_bid = compute_bid(max_other_urgency, sampled_idx);
        return self_bid + SOCIAL_BID_WEIGHT * other_bid;
    }

    double dist_sq(int i, int j) {
        double d1 = particles[i].first - particles[j].first;
        double d2 = particles[i].second - particles[j].second;
        return d1*d1 + d2*d2;
    }

    void update_weights(int sampled_idx, double surplus) {
        vector<pair<double, int>> neighbors;
        for (int i = 0; i < SMALL_N; ++i) {
            if (i == sampled_idx) continue;
            neighbors.push_back({dist_sq(sampled_idx, i), i});
        }
        sort(neighbors.begin(), neighbors.end());
        vector<double> local_reward(SMALL_N, 0.0);
        local_reward[sampled_idx] = surplus;
        for (int k = 0; k < min(K_NEIGHBORS, (int)neighbors.size()); ++k) {
            int idx = neighbors[k].second;
            double similarity = exp(-neighbors[k].first / (2.0 * EPSILON_ENTROPY));
            local_reward[idx] = surplus * similarity;
        }
        const double H_STEP = 1.0;
        for (int i = 0; i < SMALL_N; ++i) {
            if (abs(local_reward[i]) > 1e-9) {
                weights[i] *= exp(H_STEP * local_reward[i] / EPSILON_ENTROPY);
            }
        }
        double sum_w = accumulate(weights.begin(), weights.end(), 0.0);
        if (sum_w < EPS) sum_w = 1.0;
        for (auto& w : weights) {
            w = max(w / sum_w, 1e-6);
        }
        sum_w = accumulate(weights.begin(), weights.end(), 0.0);
        for (auto& w : weights) w /= sum_w;
    }
};

vector<BiddingAgent> user_agents;

// ==================== Power Optimization ====================
double power_cost(double p) { return p * log(p + 1.0); }

pair<double, double> optimize_joint_power_grid(
    int r, int k, int t,
    int u1, int u2,
    double rem1, double rem2,
    double budget) {
    const int STEPS = 20;
    double best_p1 = 0.0, best_p2 = 0.0;
    double best_score = -1e9;

    for (int i = 0; i <= STEPS; ++i) {
        for (int j = 0; j <= STEPS - i; ++j) {
            double p1 = (i * budget) / STEPS;
            double p2 = (j * budget) / STEPS;
            if (p1 + p2 > budget + EPS) continue;

            double old1 = p[r][k][t][u1], old2 = p[r][k][t][u2];
            p[r][k][t][u1] = p1;
            p[r][k][t][u2] = p2;
            update_cross_cell_power(r, t);

            double rate1 = compute_actual_rate(u1, k, t, r);
            double rate2 = compute_actual_rate(u2, k, t, r);

            double score = (rate1 / max(1.0, rem1)) + (rate2 / max(1.0, rem2))
                         - 0.01 * (power_cost(p1) + power_cost(p2));

            if (score > best_score) {
                best_score = score;
                best_p1 = p1;
                best_p2 = p2;
            }

            p[r][k][t][u1] = old1;
            p[r][k][t][u2] = old2;
        }
    }
    return {best_p1, best_p2};
}

double evaluate_power_allocation_sa(
    const vector<double>& powers, int r, int k, int t, int selected_user, double rem_bits, double budget) {
    double total_p = 0.0;
    for (double p_val : powers) total_p += p_val;
    if (total_p > budget + EPS) return -1e9;
    double numerator = s0[r][k][t][selected_user] * powers[selected_user];
    for (int m = 0; m < N; ++m) {
        if (m != selected_user && powers[m] > EPS) {
            numerator *= exp(d[m][r][k][selected_user]);
        }
    }
    double inter = cross_cell_power[r][k][t];
    for (int m = 0; m < N; ++m) {
        if (m != selected_user && powers[m] > EPS) {
            inter += s0[r][k][t][m] * powers[m];
        }
    }
    double sinr = numerator / (1.0 + inter);
    double rate = (sinr > EPS) ? W * log2(1.0 + sinr) : 0.0;
    double efficiency = rate / max(1.0, rem_bits);
    double penalty = power_cost(powers[selected_user]);
    return efficiency - 0.01 * penalty;
}

double simulated_annealing_optimize(
    int r, int k, int t, int selected_user, double rem_bits, double budget) {
    vector<double> best_powers(N, 0.0);
    double best_fitness = -1e9;
    double init_power = budget * 0.5;
    vector<double> current_powers(N, 0.0);
    current_powers[selected_user] = init_power;
    double current_fitness = evaluate_power_allocation_sa(current_powers, r, k, t, selected_user, rem_bits, budget);
    best_powers = current_powers;
    best_fitness = current_fitness;
    double temp = INITIAL_TEMP;
    auto now = chrono::steady_clock::now().time_since_epoch().count();
    mt19937 rng(static_cast<unsigned int>(now & 0xFFFFFFFF));
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int iter = 0; iter < SA_ITERATIONS && temp > MIN_TEMP; ++iter) {
        vector<double> new_powers = current_powers;
        double delta = (dist(rng) - 0.5) * 0.2 * budget;
        new_powers[selected_user] = max(0.0, min(budget, new_powers[selected_user] + delta));
        double new_fitness = evaluate_power_allocation_sa(new_powers, r, k, t, selected_user, rem_bits, budget);
        if (new_fitness > current_fitness || dist(rng) < exp((new_fitness - current_fitness) / temp)) {
            current_powers = new_powers;
            current_fitness = new_fitness;
            if (new_fitness > best_fitness) {
                best_fitness = new_fitness;
                best_powers = new_powers;
            }
        }
        temp *= COOLING_RATE;
    }
    return best_powers[selected_user];
}

// ==================== Scheduling Core ====================
void multi_agent_schedule_tti(int t) {
    for (int r = 0; r < RBG_COUNT; ++r) {
        update_cross_cell_power(r, t);
    }

    vector<double> user_urgency(N, -1e9);
    vector<int> user_best_frame(N, -1);
    for (int j : active_frames_per_tti[t]) {
        if (frame_state[j].finished) continue;
        double u_val = calculate_urgency(j, t);
        int user_id = frames[j].user;
        if (u_val > user_urgency[user_id]) {
            user_urgency[user_id] = u_val;
            user_best_frame[user_id] = j;
        }
    }

    struct Candidate {
        bool is_pair = false;
        int user1 = -1, user2 = -1;
        int frame1 = -1, frame2 = -1;
        double bid = -1e9;
        int sampled_idx1 = -1, sampled_idx2 = -1;
    };

    for (int r = 0; r < RBG_COUNT; ++r) {
        for (int k = 0; k < K; ++k) {
            double budget = 1.0 - rbg_power[r][k][t];
            if (budget <= EPS) continue;

            vector<Candidate> candidates;

            // 单用户候选
            for (int n = 0; n < N; ++n) {
                int j = user_best_frame[n];
                if (j == -1 || !is_feasible(j, t, r, k)) continue;
                double self_u = user_urgency[n];
                double max_other = -1e9;
                for (int m = 0; m < N; ++m) {
                    if (m != n && user_urgency[m] > max_other) {
                        max_other = user_urgency[m];
                    }
                }
                int sampled_idx;
                double bid = user_agents[n].compute_social_bid(self_u, max_other, sampled_idx);
                candidates.push_back({
                    .is_pair=false, .user1=n, .user2=-1,
                    .frame1=j, .frame2=-1,
                    .bid=bid, .sampled_idx1=sampled_idx, .sampled_idx2=-1
                });
            }

            // 双用户候选
            vector<pair<double, int>> urg_pairs;
            for (int n = 0; n < N; ++n) {
                if (user_best_frame[n] != -1 && is_feasible(user_best_frame[n], t, r, k)) {
                    urg_pairs.push_back({user_urgency[n], n});
                }
            }
            sort(urg_pairs.rbegin(), urg_pairs.rend());
            if (urg_pairs.size() >= 2) {
                int u1 = urg_pairs[0].second, u2 = urg_pairs[1].second;
                int j1 = user_best_frame[u1], j2 = user_best_frame[u2];
                double self1 = user_urgency[u1], self2 = user_urgency[u2];
                int idx1, idx2;
                double bid1 = user_agents[u1].compute_bid(self1, idx1);
                double bid2 = user_agents[u2].compute_bid(self2, idx2);
                candidates.push_back({
                    .is_pair=true, .user1=u1, .user2=u2,
                    .frame1=j1, .frame2=j2,
                    .bid=bid1 + bid2, .sampled_idx1=idx1, .sampled_idx2=idx2
                });
            }

            if (candidates.empty()) continue;

            auto best = *max_element(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) { return a.bid < b.bid; });

            if (!best.is_pair) {
                double rem_bits = frames[best.frame1].TBS - frame_state[best.frame1].accumulated_bits;
                double optimal_power = simulated_annealing_optimize(
                    r, k, t, best.user1, rem_bits, budget);
                p[r][k][t][best.user1] = optimal_power;
                rbg_power[r][k][t] += optimal_power;
                last_scheduled[best.user1] = t;
            } else {
                double rem1 = frames[best.frame1].TBS - frame_state[best.frame1].accumulated_bits;
                double rem2 = frames[best.frame2].TBS - frame_state[best.frame2].accumulated_bits;
                auto [p1, p2] = optimize_joint_power_grid(r, k, t, best.user1, best.user2, rem1, rem2, budget);
                p[r][k][t][best.user1] = p1;
                p[r][k][t][best.user2] = p2;
                rbg_power[r][k][t] += p1 + p2;
                last_scheduled[best.user1] = t;
                last_scheduled[best.user2] = t;
                // ✅ 已删除：不再记录 global_collab_log
            }
        }
    }

    for (int r = 0; r < RBG_COUNT; ++r) {
        update_cross_cell_power(r, t);
    }

    for (int j : active_frames_per_tti[t]) {
        if (frame_state[j].finished) continue;
        double total_rate = 0.0;
        for (int k = 0; k < K; ++k) {
            for (int r = 0; r < RBG_COUNT; ++r) {
                total_rate += compute_actual_rate(frames[j].user, k, t, r);
                frame_state[j].total_power_used += p[r][k][t][frames[j].user];
            }
        }
        frame_state[j].accumulated_bits += total_rate;
        if (frame_state[j].accumulated_bits >= frames[j].TBS - EPS) {
            frame_state[j].finished = true;
        }
    }

    vector<bool> user_success_this_tti(N, false);
    for (int j : active_frames_per_tti[t]) {
        if (frames[j].t1 == t && frame_state[j].finished) {
            user_success_this_tti[frames[j].user] = true;
        }
    }

    for (int j : active_frames_per_tti[t]) {
        if (frames[j].t1 != t || frame_state[j].updated_beta) continue;
        int u = frames[j].user;
        double surplus = frame_state[j].finished ? SURPLUS_FINISHED : SURPLUS_UNFINISHED;
        surplus -= POWER_PENALTY_COEF * frame_state[j].total_power_used;

        for (int m = 0; m < N; ++m) {
            if (m != u && user_success_this_tti[m] && user_urgency[m] > 1.0) {
                surplus += SOCIAL_BONUS;
            }
        }

        int dummy_idx = 0;
        user_agents[u].update_weights(dummy_idx, surplus);
        frame_state[j].updated_beta = true;
    }
}

// ==================== Main ====================
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    cin >> N >> K >> T >> RBG_COUNT;
    s0.assign(RBG_COUNT, vector<vector<vector<double>>>(K, vector<vector<double>>(T, vector<double>(N))));
    d.assign(N, vector<vector<vector<double>>>(RBG_COUNT, vector<vector<double>>(K, vector<double>(N))));
    p.assign(RBG_COUNT, vector<vector<vector<double>>>(K, vector<vector<double>>(T, vector<double>(N, 0.0))));
    rbg_power.assign(RBG_COUNT, vector<vector<double>>(K, vector<double>(T, 0.0)));
    cross_cell_power.assign(RBG_COUNT, vector<vector<double>>(K, vector<double>(T, 0.0)));
    last_scheduled.assign(N, -GRACE_PERIOD);
    frame_state.clear();
    active_frames_per_tti.assign(T, {});

    for (int r = 0; r < RBG_COUNT; ++r)
        for (int k = 0; k < K; ++k)
            for (int t_idx = 0; t_idx < T; ++t_idx)
                for (int n = 0; n < N; ++n)
                    cin >> s0[r][k][t_idx][n];

    for (int m = 0; m < N; ++m)
        for (int r = 0; r < RBG_COUNT; ++r)
            for (int k = 0; k < K; ++k)
                for (int n = 0; n < N; ++n)
                    cin >> d[m][r][k][n];

    int J; cin >> J;
    frames.resize(J); frame_state.resize(J);
    for (int j = 0; j < J; ++j) {
        int id; double TBS; int user, t0, td;
        cin >> id >> TBS >> user >> t0 >> td;
        frames[j] = {user, t0, t0 + td - 1, TBS};
        for (int t = frames[j].t0; t <= frames[j].t1; ++t) {
            active_frames_per_tti[t].push_back(j);
        }
    }

    user_agents.resize(N);

    for (int t = 0; t < T; ++t) {
        multi_agent_schedule_tti(t);
    }

    cout << fixed << setprecision(6);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < RBG_COUNT; ++r)
                for (int n = 0; n < N; ++n)
                    cout << (n ? " " : "") << max(0.0, min(4.0, p[r][k][t][n])) << (n == N - 1 ? "\n" : "");

    return 0;

}
