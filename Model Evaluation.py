#!/usr/bin/env python3


#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpy
from simpy.core import EmptySchedule

from ppo_agent import PPOAgent
from MM3_tandem import TandemQueue
from run_config import SIM_CONFIG, TRAIN_CONFIG, NETWORK_CONFIG, EVAL_CONFIG


# ===================== utilities =====================

def _safe_p90(values):
    s = pd.Series(values).dropna()
    return float(np.percentile(s.values, 90)) if len(s) else np.nan


def build_shift_offsets(ev_df, shifts_per_run):
    """
    Compute per-run shift start offsets (global timeline) and mean boundaries.
    Returns:
      run_offsets: dict[run][shift] -> offset (start time of that shift in global run timeline)
      mean_boundaries: [0, mean end of shift1, mean end of shift2, ..., total mean horizon]
    """
    if ev_df.empty:
        return {}, [0.0]

    # Prefer explicit end markers; fall back to max time within each (run,shift)
    ends = (ev_df[ev_df["event_type"] == "Simulation_Complete"]
              .groupby(["run","shift"])["time"]
              .max()
              .unstack())
    alt_ends = (ev_df.groupby(["run","shift"])["time"].max().unstack())
    ends = ends.combine_first(alt_ends)  # DataFrame: index=run, columns=shift

    run_offsets = {}
    durations_per_run = {}
    for run, row in ends.iterrows():
        durations = [float(row.get(s, np.nan)) for s in range(1, shifts_per_run + 1)]
        durations_per_run[run] = durations

        offsets = {1: 0.0}
        acc = 0.0
        for s in range(2, shifts_per_run + 1):
            prev = durations[s - 2]
            acc += 0.0 if np.isnan(prev) else float(prev)
            offsets[s] = acc
        run_offsets[run] = offsets

    # Mean boundaries from mean durations
    mean_durs = []
    for s in range(1, shifts_per_run + 1):
        vals = [durations_per_run[r][s-1]
                for r in durations_per_run
                if not np.isnan(durations_per_run[r][s-1])]
        mean_durs.append(float(np.mean(vals)) if vals else 0.0)

    mean_boundaries = [0.0]
    acc = 0.0
    for d in mean_durs:
        acc += d
        mean_boundaries.append(acc)

    return run_offsets, mean_boundaries


def draw_shift_boundaries(ax, boundaries):
    """Draw vertical dashed lines at provided boundary positions (including 0 and end)."""
    for x in boundaries:
        ax.axvline(x, linestyle="--", linewidth=1.0, alpha=0.5)


# ===================== global-time binnings (avg over runs) =====================

def decision_bins_global(dec_df, run_offsets, global_horizon, bin_size=0.1):
    """
    Decision-epoch metrics vs GLOBAL CURRENT TIME.
    Steps:
      - compute global_time = local time + run_offsets[run][shift]
      - per run: bin by global_time, compute stats
      - average bins across runs (equal weight per run)
    Returns averaged DataFrame with columns:
      [global_time, p_action0, q_value_0, q_value_1, q1_length, q2_length]
    """
    if dec_df.empty:
        return pd.DataFrame()

    df = dec_df.copy()
    df["global_time"] = df.apply(
        lambda r: float(r["time"]) + float(run_offsets.get(r["run"], {}).get(r["shift"], 0.0)), axis=1
    )
    tb = np.arange(0.0, float(global_horizon) + bin_size, bin_size)
    df["time_bin"] = pd.cut(df["global_time"], bins=tb, labels=tb[:-1], include_lowest=True)
    df = df.dropna(subset=["time_bin"]).copy()
    df["time_bin"] = df["time_bin"].astype(float)

    per_run = (df.groupby(["run","time_bin"], observed=True)
                 .agg(p_action0=("action", lambda x: (x == 0).mean()),
                      q_value_0=("q_value_0","mean"),
                      q_value_1=("q_value_1","mean"),
                      q1_length=("q1_length","mean"),
                      q2_length=("q2_length","mean"))
                 .reset_index())

    avg = (per_run.groupby("time_bin", observed=True)[
            ["p_action0","q_value_0","q_value_1","q1_length","q2_length"]
          ].mean()
          .reset_index()
          .rename(columns={"time_bin":"global_time"}))
    return avg


def all_actions_bins_global(ev_df, run_offsets, global_horizon, bin_size=0.1):
    """
    ALL actions = S1_Start/S2_Start vs GLOBAL CURRENT TIME.
    Returns averaged DataFrame with columns:
      [global_time, action0_prob, action1_prob, q1_mean, q2_mean]
    """
    if ev_df.empty:
        return pd.DataFrame()

    use = ev_df[ev_df["event_type"].isin(["S1_Start","S2_Start"])].copy()
    if use.empty:
        return pd.DataFrame()

    use["action"] = np.where(use["event_type"] == "S1_Start", 0, 1)
    use["global_time"] = use.apply(
        lambda r: float(r["time"]) + float(run_offsets.get(r["run"], {}).get(r["shift"], 0.0)), axis=1
    )

    tb = np.arange(0.0, float(global_horizon) + bin_size, bin_size)
    use["time_bin"] = pd.cut(use["global_time"], bins=tb, labels=tb[:-1], include_lowest=True)
    use = use.dropna(subset=["time_bin"]).copy()
    use["time_bin"] = use["time_bin"].astype(float)

    per_run = (use.groupby(["run","time_bin"], observed=True)
                 .agg(n_actions=("action","size"),
                      action0_count=("action", lambda x: (x == 0).sum()),
                      q1_mean=("q1_length","mean"),
                      q2_mean=("q2_length","mean"))
                 .reset_index())
    per_run["action0_prob"] = per_run["action0_count"] / per_run["n_actions"]
    per_run["action1_prob"] = 1.0 - per_run["action0_prob"]

    avg = (per_run.groupby("time_bin", observed=True)[
            ["action0_prob","action1_prob","q1_mean","q2_mean"]
          ].mean()
          .reset_index()
          .rename(columns={"time_bin":"global_time"}))
    return avg


def patient_metrics_global_by_arrival(ev_df, run_offsets, global_horizon, bin_size=0.1):
    """
    Patient metrics vs GLOBAL ARRIVAL TIME (arrival → downstream times can span shifts).
    Averaged across runs only.
    Returns averaged DataFrame with columns:
      [global_time, tpia_mean, tpia_p90, los_mean, los_p90, s1_to_s2_start_mean, s1_to_s2_start_p90]
    """
    if ev_df.empty:
        return pd.DataFrame()

    keep = {"Arrival","S1_Start","S1_Complete","S2_Start","S2_Complete"}
    df = ev_df[ev_df["event_type"].isin(keep)].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (run, cid), g in df.groupby(["run","customer_id"], sort=False):
        # Arrival is required
        g_arr = g[g["event_type"] == "Arrival"]
        if g_arr.empty:
            continue
        arr_row = g_arr.sort_values("time").iloc[0]
        arr_shift = int(arr_row["shift"])
        arr_global = float(arr_row["time"]) + float(run_offsets.get(run, {}).get(arr_shift, 0.0))

        def first_global(event_name):
            sub = g[g["event_type"] == event_name]
            if sub.empty:
                return None
            r = sub.sort_values("time").iloc[0]
            sh = int(r["shift"])
            return float(r["time"]) + float(run_offsets.get(run, {}).get(sh, 0.0))

        s1_start_g = first_global("S1_Start")
        s1_comp_g  = first_global("S1_Complete")
        s2_start_g = first_global("S2_Start")
        s2_comp_g  = first_global("S2_Complete")

        tpia = (s1_start_g - arr_global) if (s1_start_g is not None) else np.nan
        if s2_comp_g is not None:
            los = s2_comp_g - arr_global
        elif s1_comp_g is not None:
            los = s1_comp_g - arr_global
        else:
            los = np.nan
        s1_to_s2 = (s2_start_g - s1_comp_g) if (s1_comp_g is not None and s2_start_g is not None) else np.nan

        rows.append({
            "run": run,
            "arrival_global": arr_global,
            "tpia": tpia,
            "total_los": los,
            "s1_to_s2_start": s1_to_s2
        })

    if not rows:
        return pd.DataFrame()

    pat = pd.DataFrame(rows)
    tb = np.arange(0.0, float(global_horizon) + bin_size, bin_size)
    pat["arrival_bin"] = pd.cut(pat["arrival_global"], bins=tb, labels=tb[:-1], include_lowest=True)
    pat = pat.dropna(subset=["arrival_bin"]).copy()
    pat["arrival_bin_time"] = pat["arrival_bin"].astype(float)

    per_run = (pat.groupby(["run","arrival_bin_time"], observed=True)
                 .agg(tpia_mean=("tpia","mean"),
                      tpia_p90 =("tpia", _safe_p90),
                      los_mean =("total_los","mean"),
                      los_p90  =("total_los", _safe_p90),
                      s1_to_s2_start_mean=("s1_to_s2_start","mean"),
                      s1_to_s2_start_p90 =("s1_to_s2_start", _safe_p90))
                 .reset_index())

    avg = (per_run.groupby("arrival_bin_time", observed=True)[
            ["tpia_mean","tpia_p90","los_mean","los_p90",
             "s1_to_s2_start_mean","s1_to_s2_start_p90"]
          ].mean()
          .reset_index()
          .rename(columns={"arrival_bin_time":"global_time"}))
    return avg


# ===================== run one shift (with carryover) =====================

def seedable_env(agent, carryover, next_id_start):
    init_q1 = carryover.get('q1', [])
    init_q2 = carryover.get('q2_by_server', [[] for _ in range(SIM_CONFIG['num_servers'])])
    env = TandemQueue(
        simpy.Environment(),
        agent=agent,
        is_training=True,
        initial_queue1=init_q1,
        initial_q2_by_server=init_q2,
        next_customer_id_start=next_id_start
    )
    return env


def extract_carryover(env: TandemQueue):
    q1 = []
    for c in list(env.queue1):
        abandon_remaining = None
        if getattr(c, 'abandon_deadline', None) is not None:
            abandon_remaining = float(c.abandon_deadline) - float(env.env.now)

        # Skip anyone who should have already abandoned
        if abandon_remaining is not None and abandon_remaining <= 0:
            continue

        q1.append({
            'id': c.id,
            'arrival_time': c.arrival_time,
            'start_service_1_time': c.start_service_1_time,
            'W1': c.W1, 'T1': c.T1,
            'start_service_2_time': c.start_service_2_time,
            'W2': c.W2, 'T2': c.T2,
            'abandon_remaining': abandon_remaining if abandon_remaining is None else float(abandon_remaining)
        })
    # Q2 waiting per server
    q2_by_server = []
    for srv in env.servers:
        lst = []
        for c in list(srv.queue2):
            lst.append({
                'id': c.id,
                'arrival_time': c.arrival_time,
                'start_service_1_time': c.start_service_1_time,
                'W1': c.W1, 'T1': c.T1,
                'start_service_2_time': c.start_service_2_time,
                'W2': c.W2, 'T2': c.T2
            })
        q2_by_server.append(lst)

    all_ids = [rec['id'] for rec in q1] + [rec['id'] for sub in q2_by_server for rec in sub]
    next_id_start = (max(all_ids) + 1) if all_ids else env.next_customer_id
    return {'q1': q1, 'q2_by_server': q2_by_server}, next_id_start


def run_one_shift(agent, run_id, shift_id, carryover, next_id_start):
    env = seedable_env(agent, carryover, next_id_start)
    # bootstrap to first decision epoch (or finish)
    try:
        ticks = 0
        while (env.current_decision_state is None) and (not env.is_simulation_complete()):
            env.env.step()
            if hasattr(env, 'try_forced_service_on_idle_servers'):
                env.try_forced_service_on_idle_servers()
            ticks += 1
            if ticks > 1_000_000:
                raise RuntimeError("Bootstrap stuck (no decision epoch reached).")
    except EmptySchedule:
        carry_next, next_id_start = extract_carryover(env)
        return [], [], 0, carry_next, next_id_start

    if env.current_decision_state is None:
        carry_next, next_id_start = extract_carryover(env)
        return [], [], 0, carry_next, next_id_start

    state = env.get_state_array(env.current_decision_state)
    decisions, all_events, agent_choice_count = [], [], 0

    while not env.is_simulation_complete():
        # map state
        mtype = TRAIN_CONFIG.get('modified_states_type', None)
        if mtype == 'Q1Q2s':
            state_mod = state[:-2]
            q1_len = int(state[0]); sid = int(state[-2]); q2_len = int(state[1 + sid]); t = float(state[-1])
        elif mtype == 'Q1Q2':
            state_mod = state
            q1_len = int(state[0]); q2_len = int(state[1]); t = float(state[2])
        else:
            state_mod = state
            q1_len = int(state[0]); sid = int(state[-2]); q2_len = int(state[1 + sid]); t = float(state[-1])

        # action selection with capacity rules
        q2_cap = (q2_len >= env.q2_capacity)
        if q2_cap and q2_len > 0:
            action = 1; a_type = "FORCED Q2 (capacity)"
        elif q1_len > 0 and q2_len == 0:
            action = 0; a_type = "FORCED Q1"
        elif q1_len == 0 and q2_len > 0:
            action = 1; a_type = "FORCED Q2"
        elif q1_len == 0 and q2_len == 0:
            action = 0; a_type = "WAITING (dummy)"
        else:
            if q2_cap:
                action = 1; a_type = "FORCED Q2 (capacity)"
            else:
                action = agent.select_action(state_mod); a_type = "AGENT CHOICE"

        # Get action probabilities from PPO agent
        action_probs = agent.get_action_probabilities(state_mod)
        q_value_0, q_value_1 = action_probs[0], action_probs[1]
            
        if a_type == "AGENT CHOICE":
            agent_choice_count += 1

        decisions.append({
            'run': run_id,
            'shift': shift_id,
            'time': t,
            'q1_length': q1_len,
            'q2_length': q2_len,
            'action': action,
            'action_type': a_type,
            'q_value_0': q_value_0,
            'q_value_1': q_value_1,
            'reward': 0
        })

        next_state, reward, done, _ = env.step(action, state_mod)
        decisions[-1]['reward'] = reward
        if not done:
            state = next_state
        else:
            break

    # collect all events
    for ev in env.all_events:
        all_events.append({
            'run': run_id,
            'shift': shift_id,
            'time': ev['time'],
            'q1_length': ev.get('q1_length', 0),
            'q2_length': sum(ev.get('q2_lengths', [0])),
            'event_type': ev['event_type'],
            'customer_id': ev.get('customer_id', None),
            'server_id': ev.get('server_id', None)
        })

    carry_next, next_id_start = extract_carryover(env)
    return decisions, all_events, agent_choice_count, carry_next, next_id_start


# ===================== main =====================

def main():
    print("=== Multi-Shift Evaluation (averaged over runs; global time) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_runs = EVAL_CONFIG['num_runs']
    shifts_per_run = EVAL_CONFIG['num_episodes']   # shifts per run

    # agent dims
    if TRAIN_CONFIG['modified_states_type'] == 'Q1Q2s':
        state_dim = 2
    elif TRAIN_CONFIG['modified_states_type'] == 'Q1Q2':
        state_dim = 3
    else:
        state_dim = 4
    action_dim = NETWORK_CONFIG['action_dim']

    # Create PPO agent
    print(f"Creating PPO agent for evaluation...")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    model_path = "trained_models/ppo_trained_model.pth"

    # Load PPO model
    if not os.path.exists(model_path):
        print(f"Error: PPO model file not found at {model_path}")
        return
    
    ckpt = torch.load(model_path, map_location=device)
    
    # PPO model loading
    if 'network_state_dict' in ckpt:
        agent.network.load_state_dict(ckpt['network_state_dict'])
        print("Loaded PPO agent checkpoint")
    else:
        print("Error: Invalid PPO model format")
        return

    # run loops
    all_decisions, all_events = [], []
    total_agent_choices = 0

    for run in range(1, num_runs + 1):
        carry = {'q1': [], 'q2_by_server': [[] for _ in range(SIM_CONFIG['num_servers'])]}
        next_id_start = 0
        print(f"\n--- Run {run}/{num_runs} ---")
        for sh in range(1, shifts_per_run + 1):
            dec, evs, choices, carry, next_id_start = run_one_shift(
                agent, run, sh, carry, next_id_start
            )
            all_decisions.extend(dec)
            all_events.extend(evs)
            total_agent_choices += choices
            print(f"  Shift {sh}/{shifts_per_run}: decisions={len(dec)}, events={len(evs)}, "
                  f"carry_q1={len(carry['q1'])}, carry_q2={sum(len(x) for x in carry['q2_by_server'])}")

    print("\nAll runs complete.")
    print(f"Total agent choices: {total_agent_choices}")

    # DataFrames
    os.makedirs('output', exist_ok=True)
    dec_df = pd.DataFrame(all_decisions)
    ev_df  = pd.DataFrame(all_events)

    if not dec_df.empty:
        dec_df.to_csv('output/decisions_runs_raw.csv', index=False)
        print(f"Saved raw decisions to output/decisions_runs_raw.csv ({len(dec_df)})")
    if not ev_df.empty:
        ev_df.to_csv('output/events_runs_raw.csv', index=False)
        print(f"Saved raw events to output/events_runs_raw.csv ({len(ev_df)})")

    # Build shift offsets (per run) + mean boundaries (for plotting)
    run_offsets, mean_boundaries = build_shift_offsets(ev_df, shifts_per_run=shifts_per_run)
    global_horizon = float(mean_boundaries[-1])

    # Global-time, averaged-over-runs metrics
    dec_avg  = decision_bins_global(dec_df,  run_offsets, global_horizon, bin_size=0.1) if not dec_df.empty else pd.DataFrame()
    act_avg  = all_actions_bins_global(ev_df, run_offsets, global_horizon, bin_size=0.1) if not ev_df.empty else pd.DataFrame()
    pat_avg  = patient_metrics_global_by_arrival(ev_df, run_offsets, global_horizon, bin_size=0.1) if not ev_df.empty else pd.DataFrame()

    if not dec_avg.empty:
        dec_avg["p_action1"] = 1.0 - dec_avg["p_action0"]
        dec_avg.to_csv("output/decision_bins_GLOBAL_avg_over_runs.csv", index=False)
    if not act_avg.empty:
        act_avg.to_csv("output/all_actions_bins_GLOBAL_avg_over_runs.csv", index=False)
    if not pat_avg.empty:
        pat_avg.to_csv("output/patient_metrics_GLOBAL_avg_over_runs.csv", index=False)

    # ---------------- plots (global time; vertical mean boundaries) ----------------
    # Decision epochs
    if not dec_avg.empty:
        fig1, (ax1, ax2) = plt.subplots(2,1, figsize=(12,10))
        t = dec_avg["global_time"]
        ax1.plot(t, dec_avg["p_action0"], '-', color='orange', linewidth=2, label='P(action=0)')
        ax1.plot(t, dec_avg["p_action1"], '-', color='blue', linewidth=2, label='P(action=1)')
        ax1.set_title("Decision-epoch probabilities (avg over runs, global time)")
        ax1.set_xlabel("Global time"); ax1.set_ylabel("Probability"); ax1.grid(True, alpha=0.3); ax1.legend(); ax1.set_ylim(0,1)
        draw_shift_boundaries(ax1, mean_boundaries)

        ax2.plot(t, dec_avg["q1_length"], '-', color='orange', linewidth=2, label="$l_0$ mean")
        ax2.plot(t, dec_avg["q2_length"], '-', color='blue', linewidth=2, label="$l_1$ mean")
        ax2.set_title("Queue lengths at decision epochs (avg over runs, global time)")
        ax2.set_xlabel("Global time"); ax2.set_ylabel("Avg length"); ax2.grid(True, alpha=0.3); ax2.legend()
        draw_shift_boundaries(ax2, mean_boundaries)

        plt.tight_layout(); plt.savefig("output/decisions_actions_avg_over_runs.png", dpi=300, bbox_inches='tight'); plt.show()

    # ALL actions
    if not act_avg.empty:
        fig2, (bx1, bx2) = plt.subplots(2,1, figsize=(12,10))
        tA = act_avg["global_time"]
        bx1.plot(tA, act_avg["action0_prob"], '-', color='orange', linewidth=2, label='P(action=0)')
        bx1.plot(tA, act_avg["action1_prob"], '-', color='blue', linewidth=2, label='P(action=1)')
        bx1.set_title("ALL-action probabilities (avg over runs, global time)")
        bx1.set_xlabel("Global time"); bx1.set_ylabel("Probability"); bx1.grid(True, alpha=0.3); bx1.legend(); bx1.set_ylim(0,1)
        draw_shift_boundaries(bx1, mean_boundaries)

        bx2.plot(tA, act_avg["q1_mean"], '-', color='orange', linewidth=2, label="$l_0$ mean")
        bx2.plot(tA, act_avg["q2_mean"], '-', color='blue', linewidth=2, label="$l_1$ mean")
        bx2.set_title("Queue lengths at ALL action starts (avg over runs, global time)")
        bx2.set_xlabel("Global time"); bx2.set_ylabel("Avg length"); bx2.grid(True, alpha=0.3); bx2.legend()
        draw_shift_boundaries(bx2, mean_boundaries)

        plt.tight_layout(); plt.savefig("output/all_actions_avg_over_runs.png", dpi=300, bbox_inches='tight'); plt.show()

    # Patient metrics (x = GLOBAL arrival time)
    if not pat_avg.empty:
        fig3, (px1, px2, px3) = plt.subplots(3,1, figsize=(12,12))
        tP = pat_avg["global_time"]

        px1.plot(tP, pat_avg["tpia_mean"], '-', linewidth=2, label='mean')
        px1.plot(tP, pat_avg["tpia_p90"],  '--', linewidth=1.5, label='p90')
        px1.set_title("TPIA vs global arrival time (avg over runs)")
        px1.set_xlabel("Global time"); px1.set_ylabel("TPIA"); px1.grid(True, alpha=0.3); px1.legend()
        draw_shift_boundaries(px1, mean_boundaries)

        px2.plot(tP, pat_avg["los_mean"], '-', linewidth=2, label='mean')
        px2.plot(tP, pat_avg["los_p90"],  '--', linewidth=1.5, label='p90')
        px2.set_title("LOS vs global arrival time (avg over runs)")
        px2.set_xlabel("Global time"); px2.set_ylabel("LOS"); px2.grid(True, alpha=0.3); px2.legend()
        draw_shift_boundaries(px2, mean_boundaries)

        px3.plot(tP, pat_avg["s1_to_s2_start_mean"], '-', linewidth=2, label='mean')
        px3.plot(tP, pat_avg["s1_to_s2_start_p90"],  '--', linewidth=1.5, label='p90')
        px3.set_title("S1→S2 vs global arrival time (avg over runs)")
        px3.set_xlabel("Global time"); px3.set_ylabel("S1→S2"); px3.grid(True, alpha=0.3); px3.legend()
        draw_shift_boundaries(px3, mean_boundaries)

        plt.tight_layout(); plt.savefig("output/patient_metrics_avg_over_runs.png", dpi=300, bbox_inches='tight'); plt.show()


if __name__ == "__main__":
    main()
