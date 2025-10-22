import simpy
import random
from collections import deque
import pandas as pd
import os
import numpy as np
from typing import Dict, List, Optional
from run_config import SIM_CONFIG, TRAIN_CONFIG, REPLAY_BUFFER_CONFIG
from replay_buffer import ReplayBuffer


class TandemQueue:
    def __init__(self, env, agent=None, **kwargs):
        self.env = env
        self.num_servers = kwargs.get('num_servers', SIM_CONFIG['num_servers'])
        self.arrival_rate = kwargs.get('arrival_rate', SIM_CONFIG['arrival_rate'])
        self.service_rate_1 = kwargs.get('service_rate_1', SIM_CONFIG['service_rate_1'])
        self.service_rate_2 = kwargs.get('service_rate_2', SIM_CONFIG['service_rate_2'])
        self.alpha = kwargs.get('alpha', SIM_CONFIG['alpha'])
        self.beta  = kwargs.get('beta',  SIM_CONFIG['beta'])
        self.reward_type = kwargs.get('reward_type', SIM_CONFIG['reward_type'])
        self.q2_capacity = kwargs.get('q2_capacity', SIM_CONFIG['q2_capacity'])
        self.agent = agent

        # --- carryover inputs ---
        init_q1 = kwargs.get('initial_queue1', [])
        init_q2_by_server = kwargs.get('initial_q2_by_server', [[] for _ in range(self.num_servers)])
        self.next_customer_id = kwargs.get('next_customer_id_start', 0)

        # Create servers and queues
        self.servers = [Server(env, self, i) for i in range(self.num_servers)]
        self.queue1 = deque()

        # Stats
        self.system_times = {}
        self.completed_customers = {}
        self.all_events = []
        self.decision_events = []

        # Replay
        self.state_dim = REPLAY_BUFFER_CONFIG['state_dim']
        self.replay_buffer = ReplayBuffer(
            capacity=REPLAY_BUFFER_CONFIG['capacity'],
            state_dim=self.state_dim
        )

        self.is_training = kwargs.get('is_training', True)
        self.new_customer_event = self.env.event()

        # Seed carryovers BEFORE starting processes (and schedule their abandonments)
        self._seed_carryover(init_q1, init_q2_by_server)

        # Start processes
        self.env.process(self.customer_arrivals())
        self.env.process(self.patient_abandonment_process())

        self.current_decision_state = None
        self.current_decision_server = None
        self.last_decision_time = 0
        self.done = False

    # ---------------------------
    # Patience distribution helper
    # ---------------------------
    def _sample_patience(self) -> float:
        """
        Draw a patience time from SIM_CONFIG.
        Supported:
          - patience_dist: 'exponential' (default), uses 'theta' (rate)
          - 'deterministic': uses 'patience_time'
          - 'weibull': uses 'patience_k' (shape), 'patience_lambda' (scale)
          - 'gamma': uses 'patience_k' (shape), 'patience_theta' (scale)
          - 'lognormal': uses 'patience_mu', 'patience_sigma'
        Return float('inf') to indicate 'no abandonment'.
        """
        dist = SIM_CONFIG.get('patience_dist', 'exponential').lower()
        if dist == 'exponential':
            theta = float(SIM_CONFIG.get('theta', 0.0))
            return random.expovariate(theta) if theta > 0 else float('inf')
        if dist == 'deterministic':
            t = float(SIM_CONFIG.get('patience_time', 0.0))
            return t if t > 0 else float('inf')
        if dist == 'weibull':
            k = float(SIM_CONFIG.get('patience_k', 1.0))
            lam = float(SIM_CONFIG.get('patience_lambda', 1.0))
            return float(np.random.weibull(k) * lam)
        if dist == 'gamma':
            k = float(SIM_CONFIG.get('patience_k', 1.0))
            theta = float(SIM_CONFIG.get('patience_theta', 1.0))  # scale
            return float(np.random.gamma(k, theta))
        if dist == 'lognormal':
            mu = float(SIM_CONFIG.get('patience_mu', 0.0))
            sigma = float(SIM_CONFIG.get('patience_sigma', 1.0))
            return float(np.random.lognormal(mean=mu, sigma=sigma))
        # Fallback
        theta = float(SIM_CONFIG.get('theta', 0.0))
        return random.expovariate(theta) if theta > 0 else float('inf')

    def _schedule_abandonment(self, customer: 'Customer', remaining: Optional[float] = None):
        """
        Schedule (or reschedule) abandonment for a Q1 customer.
        If 'remaining' is provided (carryover), use it; else draw a fresh patience.
        """
        if remaining is None:
            remaining = self._sample_patience()
        # Skip scheduling if no abandonment (inf or non-positive)
        if not np.isfinite(remaining) or remaining <= 0:
            return
        customer.patience_remaining = remaining
        customer.abandon_deadline = self.env.now + remaining
        self.env.process(self.handle_abandonment(customer, remaining))

    # ---------------------------
    # Carryover seeding (now schedules residual abandonments)
    # ---------------------------
    def _seed_carryover(self, q1_records, q2_by_server_records):
        max_id = -1

        # Q1 waiting
        for rec in q1_records:
            c = Customer(rec['id'], rec['arrival_time'])
            c.start_service_1_time = rec.get('start_service_1_time')
            c.start_service_2_time = rec.get('start_service_2_time')
            c.W1 = rec.get('W1'); c.W2 = rec.get('W2')
            c.T1 = rec.get('T1'); c.T2 = rec.get('T2')
            # residual patience (if provided by extractor)
            rem = rec.get('abandon_remaining', rec.get('patience_remaining', None))
            self.queue1.append(c)
            self._schedule_abandonment(c, remaining=rem)
            max_id = max(max_id, c.id if c.id is not None else -1)

        # Q2 waiting per server (no abandonment in Q2)
        for sid, lst in enumerate(q2_by_server_records):
            for rec in lst:
                c = Customer(rec['id'], rec['arrival_time'])
                c.start_service_1_time = rec.get('start_service_1_time')
                c.start_service_2_time = rec.get('start_service_2_time')
                c.W1 = rec.get('W1'); c.W2 = rec.get('W2')
                c.T1 = rec.get('T1'); c.T2 = rec.get('T2')
                self.servers[sid].queue2.append(c)
                max_id = max(max_id, c.id if c.id is not None else -1)

        # ensure next_customer_id continues from the max id + 1 (or provided start)
        self.next_customer_id = max(self.next_customer_id, (max_id + 1))

        # (Optional) log a seed snapshot
        self.record_event('Shift_Start_Seed', customer_id=None, additional_info={
            'seed_q1': len(self.queue1),
            'seed_q2_by_server': [len(s.queue2) for s in self.servers]
        })

    # ---------------------------
    # Small helper: force progress
    # ---------------------------
    def try_forced_service_on_idle_servers(self):
        if self.env.now >= SIM_CONFIG['sim_time']:
            return

        has_real_agent = (self.agent is not None and not isinstance(self.agent, str))

        # Only block on an existing decision state when training with a real agent
        if self.is_training and has_real_agent and self.current_decision_state is not None:
            return

        for srv in self.servers:
            if srv.busy:
                continue
            q1 = len(self.queue1)
            q2 = len(srv.queue2)

            if q1 > 0 and q2 > 0:
                if self.is_training and has_real_agent:
                    # RL run: expose state and wait for .step()
                    self.current_decision_state  = self.get_state(srv)
                    self.current_decision_server = srv
                else:
                    # pretrain/heuristic: decide now so we don't stall
                    self.env.process(srv.decide_action1())
                return

            # Q2 capacity (forced Q2)
            if q2 >= self.q2_capacity and q2 > 0:
                self.decision_events.append({
                    'time': self.env.now,
                    'state': self.get_state(srv),   # pre-action snapshot
                    'action': 'Q2',
                    'server_id': srv.server_id,
                    'forced': True
                })
                cust = srv.queue2.popleft()
                srv.busy = True
                self.env.process(srv.process_s2(cust))
                return

            # Only Q1 has work (forced Q1)
            if q1 > 0 and q2 == 0:
                self.decision_events.append({
                    'time': self.env.now,
                    'state': self.get_state(srv),
                    'action': 'Q1',
                    'server_id': srv.server_id,
                    'forced': True
                })
                cust = self.queue1.popleft()
                srv.busy = True
                self.env.process(srv.process_s1(cust))
                return

            # Only Q2 has work (forced Q2)
            if q1 == 0 and q2 > 0:
                self.decision_events.append({
                    'time': self.env.now,
                    'state': self.get_state(srv),
                    'action': 'Q2',
                    'server_id': srv.server_id,
                    'forced': True
                })
                cust = srv.queue2.popleft()
                srv.busy = True
                self.env.process(srv.process_s2(cust))
                return


    def reset(self):
        self.initial_state = None
        while self.initial_state is None and not self.is_simulation_complete():
            self.env.step()
            self.try_forced_service_on_idle_servers()
            if self.current_decision_state is not None:
                self.initial_state = self.current_decision_state
        return None if self.initial_state is None else self.get_state_array(self.initial_state)

    def record_event(self, event_type, customer_id, server_id=None, additional_info=None):
        event = {
            'time': self.env.now,
            'event_type': event_type,
            'customer_id': customer_id,
            'server_id': server_id,
            'q1_length': len(self.queue1),
            'q2_lengths': [len(server.queue2) for server in self.servers],
            'server_status': [1 if server.busy else 0 for server in self.servers]
        }
        if additional_info:
            event.update(additional_info)
        self.all_events.append(event)

    # ---------------------------
    # Abandonment (Q1 only)
    # ---------------------------
    def patient_abandonment_process(self):
        while True:
            yield self.new_customer_event
            if self.env.now >= SIM_CONFIG['sim_time']:
                break
            # schedule for the newly arrived (last in Q1)
            if len(self.queue1) > 0:
                customer = self.queue1[-1]
                # draw patience and schedule; stores deadline inside the customer
                self._schedule_abandonment(customer, remaining=None)
            # arm next trigger
            self.new_customer_event = self.env.event()

    def handle_abandonment(self, customer, abandonment_time):
        yield self.env.timeout(abandonment_time)
        if customer in self.queue1 and self.env.now < SIM_CONFIG['sim_time']:
            self.queue1.remove(customer)
            wait_time = self.env.now - customer.arrival_time  # local; may cross shifts
            self.record_event('Abandonment', customer.id, additional_info={
                'wait_time': wait_time
            })
            self.try_forced_service_on_idle_servers()

    # ---------------------------
    # Arrivals (use self.next_customer_id)
    # ---------------------------
    def customer_arrivals(self):
        while True:
            if self.env.now >= SIM_CONFIG['sim_time']:
                self.arrival_rate = 0
                break
            if self.arrival_rate <= 0:
                break
            if len(self.queue1) >= SIM_CONFIG['q1_capacity']:
                self.arrival_rate = 0
                break

            yield self.env.timeout(random.expovariate(self.arrival_rate))
            customer = Customer(self.next_customer_id, self.env.now)
            self.next_customer_id += 1

            self.queue1.append(customer)

            if not self.new_customer_event.triggered:
                self.new_customer_event.succeed()

            self.record_event('Arrival', customer.id)

            if self.env.now < SIM_CONFIG['sim_time']:
                self.try_forced_service_on_idle_servers()

            # if self.env.now < SIM_CONFIG['sim_time'] and (self.agent is None or isinstance(self.agent, str)):
            #     for server in self.servers:
            #         if not server.busy:
            #             self.env.process(server.decide_action1())

    # ---------------------------
    # State helpers
    # ---------------------------
    def get_state(self, current_server):
        return {
            'q1_length': len(self.queue1),
            'q2_lengths': [len(server.queue2) for server in self.servers],
            'server_id': current_server.server_id,
            'time': self.env.now
        }

    def get_state_array(self, state_dict):
        if TRAIN_CONFIG.get('modified_states_type') == 'Q1Q2':
            sid = state_dict['server_id']
            q2s = state_dict['q2_lengths'][sid]
            return np.array([state_dict['q1_length'], q2s, state_dict['time']], dtype=np.float32)
        else:
            return np.array([
                state_dict['q1_length'],
                *state_dict['q2_lengths'],
                state_dict['server_id'],
                state_dict['time']
            ], dtype=np.float32)

    def save_all_events(self, filename="all_events.csv"):
        try:
            os.makedirs("output", exist_ok=True)
            df_events = pd.DataFrame(self.all_events)
            filepath = os.path.join("output", filename)
            df_events.to_csv(filepath, index=False)
            print(f"\nSaved {len(self.all_events)} events to {filepath}")
            print("\nEvent types distribution:")
            print(df_events['event_type'].value_counts())
        except Exception as e:
            print(f"Error saving events: {str(e)}")

    # ---------------------------
    # One RL step between decisions (event-driven cumulative reward)
    # ---------------------------
    def step(self, action, current_state=None):
        events = []

        if self.current_decision_state is None:
            raise ValueError("Current decision state is None. Cannot proceed with step.")

        server = self.current_decision_server
        decision_time = self.env.now
        cum_reward = 0.0

        # Decide what actually executes (respect capacity)
        q2_at_capacity = len(server.queue2) >= self.q2_capacity
        executed_action = None  # 0 = Q1, 1 = Q2

        if action == 0 and not q2_at_capacity:
            # Serve Q1 (non-forced)
            if len(self.queue1) > 0:
                customer = self.queue1.popleft()
                server.busy = True
                executed_action = 0
                self.record_event('Decision_ServeQ1', customer.id, server_id=server.server_id)
                events.append({
                    'time': self.env.now, 'event_type': 'Decision_ServeQ1',
                    'customer_id': customer.id, 'server_id': server.server_id,
                    'q1_length': len(self.queue1),
                    'q2_lengths': [len(s.queue2) for s in self.servers],
                    'servers_busy': [s.busy for s in self.servers]
                })
                # starting the process logs S1_Start immediately at this same timestamp
                self.env.process(server.process_s1(customer))
        else:
            # Serve Q2 (forced if at capacity or chosen action==1)
            if len(server.queue2) > 0:
                customer = server.queue2.popleft()
                server.busy = True
                executed_action = 1
                self.record_event('Decision_ServeQ2', customer.id, server_id=server.server_id,
                                additional_info={'forced_action': q2_at_capacity})
                events.append({
                    'time': self.env.now, 'event_type': 'Decision_ServeQ2',
                    'customer_id': customer.id, 'server_id': server.server_id,
                    'q1_length': len(self.queue1),
                    'q2_lengths': [len(s.queue2) for s in self.servers],
                    'servers_busy': [s.busy for s in self.servers],
                    'forced_action': q2_at_capacity
                })
                # starting the process logs S2_Start immediately at this same timestamp
                self.env.process(server.process_s2(customer))

        # Reward for the immediate action at the decision time (count once)
        if executed_action is not None:
            cum_reward += self.calculate_reward(decision_time, decision_time, executed_action)

        # Only accumulate rewards for start-events AFTER decision_time
        events_cursor = len(self.all_events)

        while True:
            # Advance one sim event
            self.env.step()

            # Newly logged events since last sweep
            new_events = self.all_events[events_cursor:]
            events_cursor = len(self.all_events)

            # Accumulate reward for forced starts strictly after the decision time
            for ev in new_events:
                events.append(ev)  # keep full trace
                ev_t = ev.get('time', -1)
                if ev_t <= decision_time:
                    continue
                et = ev.get('event_type')
                if et == 'S1_Start':
                    cum_reward += self.calculate_reward(ev_t, ev_t, 0)
                elif et == 'S2_Start':
                    cum_reward += self.calculate_reward(ev_t, ev_t, 1)

            # Terminal?
            if self.is_simulation_complete():
                events.append({
                    'time': self.env.now,
                    'event_type': 'Simulation_Complete',
                    'q1_length': len(self.queue1),
                    'q2_lengths': [len(s.queue2) for s in self.servers],
                    'servers_busy': [s.busy for s in self.servers]
                })
                self.current_decision_state = None
                self.current_decision_server = None
                return None, cum_reward, True, events

            # Next decision epoch? (idle server with q1>0 and q2>0)
            if self.env.now < SIM_CONFIG['sim_time']:
                for srv in self.servers:
                    if not srv.busy:
                        q1 = len(self.queue1)
                        q2 = len(srv.queue2)
                        if q1 > 0 and q2 > 0:
                            self.current_decision_state = self.get_state(srv)
                            self.current_decision_server = srv
                            events.append({
                                'time': self.env.now,
                                'event_type': 'Next_Decision_Point',
                                'server_id': srv.server_id,
                                'q1_length': len(self.queue1),
                                'q2_lengths': [len(s.queue2) for s in self.servers],
                                'servers_busy': [s.busy for s in self.servers]
                            })
                            next_state = self.get_state_array(self.current_decision_state)
                            if current_state is not None:
                                mtype = TRAIN_CONFIG.get('modified_states_type', None)
                                if mtype == 'Q1Q2s':
                                    # strip server_id and time
                                    next_state = next_state[:-2]
                            return next_state, cum_reward, False, events

            # Keep the system progressing (forced logic)
            if self.env.now < SIM_CONFIG['sim_time']:
                for srv in self.servers:
                    if not srv.busy:
                        q1 = len(self.queue1)
                        q2 = len(srv.queue2)
                        if q2 >= self.q2_capacity and q2 > 0:
                            cust = srv.queue2.popleft()   # FIFO (fixed)
                            srv.busy = True
                            self.env.process(srv.process_s2(cust))
                            break
                        elif q1 > 0 and q2 == 0:
                            cust = self.queue1.popleft()
                            srv.busy = True
                            self.env.process(srv.process_s1(cust))
                            break
                        elif q1 == 0 and q2 > 0:
                            cust = srv.queue2.popleft()
                            srv.busy = True
                            self.env.process(srv.process_s2(cust))
                            break

    # ---------------------------
    # Reward (single-event reward)
    # ---------------------------
    def calculate_reward(self, start_time, end_time, action):
        """
        Immediate reward for a single event that occurred at start_time.
        action: 1 for Q2 (S2_Start), 0 for Q1 (S1_Start).
        """
        if self.reward_type == 'physician':
            # Each S2_Start yields +1, S1_Start yields 0
            return 1.0 if action == 1 else 0.0

        elif self.reward_type == 'ED':
            # Use the first event at/after start_time to read the 'after-action' snapshot
            post = [e for e in self.all_events
                    if e['time'] >= start_time and
                    e['event_type'] in ('S1_Start', 'S2_Start', 'Decision_ServeQ1', 'Decision_ServeQ2')]
            if post:
                first = min(post, key=lambda x: x['time'])
                q1_after = int(first['q1_length'])
                q2_after = sum(int(x) for x in first['q2_lengths'])
            else:
                # very rare fallback
                q1_after = len(self.queue1)
                q2_after = sum(len(s.queue2) for s in self.servers)

            return -(self.alpha * q1_after / self.service_rate_1 +
                    self.beta  * q2_after  / self.service_rate_2)

        return 0.0




    def process_decision_events(self):
        """
        Populate self.replay_buffer with transitions between consecutive decision events.
        Uses decision states captured in Server.decide_action1() during pretraining runs.
        """
        # No decisions logged -> nothing to add
        if not self.decision_events:
            return

        def map_state(sdict):
            arr = self.get_state_array(sdict)
            mtype = TRAIN_CONFIG.get('modified_states_type', None)
            if mtype == 'Q1Q2s':
                # your convention: drop last 2 entries (server_id, time)
                return arr[:-2]
            elif mtype == 'Q1Q2':
                # already [q1, q2_server, time]
                return arr
            else:
                return arr

        # Transitions between consecutive decision epochs
        for i in range(len(self.decision_events) - 1):
            cur = self.decision_events[i]
            nxt = self.decision_events[i + 1]

            start_t = cur['time']
            end_t   = nxt['time']

            state      = map_state(cur['state'])
            next_state = map_state(nxt['state'])
            action     = 0 if cur['action'] == 'Q1' else 1

            reward = self.calculate_reward(start_t, end_t, action)
            self.replay_buffer.add(state, action, reward, next_state, end_t, False)

        # Final transition from last decision until current env time
        last = self.decision_events[-1]
        start_t = last['time']
        end_t   = self.env.now

        state  = map_state(last['state'])
        action = 0 if last['action'] == 'Q1' else 1
        reward = self.calculate_reward(start_t, end_t, action)

        # Terminal: next_state=None
        self.replay_buffer.add(state, action, reward, None, end_t, True)

    def is_simulation_complete(self):
        return (self.arrival_rate == 0 and all(not server.busy for server in self.servers))


class Server:
    def __init__(self, env, system, server_id):
        self.env = env
        self.system = system
        self.server_id = server_id
        self.queue2 = deque()
        self.busy = False

    def decide_action1(self):  ## used for pretrinaing
        if hasattr(self.system, 'agent') and self.system.agent is not None and not isinstance(self.system.agent, str):
            return
        if self.busy or self.env.now >= SIM_CONFIG['sim_time']:
            return

        if len(self.system.queue1) > 0 or len(self.queue2) > 0:
            q2_cap = (len(self.queue2) >= self.system.q2_capacity and len(self.queue2) > 0)

            # 1) choose action, but DO NOT pop yet
            if q2_cap:
                action, proc = 'Q2', self.process_s21
            elif len(self.system.queue1) > 0 and len(self.queue2) == 0:
                action, proc = 'Q1', self.process_s11
            elif len(self.system.queue1) == 0 and len(self.queue2) > 0:
                action, proc = 'Q2', self.process_s21
            else:
                action, proc = ('Q1', self.process_s11) if random.random() < 0.5 else ('Q2', self.process_s21)

            # 2) log the *true* pre-decision state (no +1 needed)
            decision_event = {
                'time': self.env.now,
                'state': {
                    'q1_length': len(self.system.queue1),
                    'q2_lengths': [len(s.queue2) for s in self.system.servers],
                    'server_id': self.server_id,
                    'time': self.env.now
                },
                'action': action,
                'server_id': self.server_id
            }
            self.system.decision_events.append(decision_event)

            # 3) now pop and start service
            if action == 'Q1':
                customer = self.system.queue1.popleft()
            else:  # 'Q2'
                customer = self.queue2.popleft()

            self.busy = True
            yield self.env.process(proc(customer))


    # --------- Service processes ---------
    def process_s11(self, customer: 'Customer'):
        self.busy = True
        service_time = random.expovariate(self.system.service_rate_1)
        customer.start_service_1_time = self.env.now
        customer.W1 = self.env.now - customer.arrival_time
        self.system.record_event('S1_Start', customer.id, self.server_id, {'service_time': service_time})
        yield self.env.timeout(service_time)
        customer.T1 = self.env.now - customer.arrival_time
        self.queue2.append(customer)
        self.busy = False
        self.system.record_event('S1_Complete', customer.id, self.server_id, {'T1': customer.T1})
        self.system.try_forced_service_on_idle_servers()

    def process_s1(self, customer: 'Customer'):
        self.busy = True
        service_time = random.expovariate(self.system.service_rate_1)
        customer.start_service_1_time = self.env.now
        customer.W1 = self.env.now - customer.arrival_time
        self.system.record_event('S1_Start', customer.id, self.server_id, {'service_time': service_time})
        yield self.env.timeout(service_time)
        customer.T1 = self.env.now - customer.arrival_time
        self.queue2.append(customer)
        self.busy = False
        self.system.record_event('S1_Complete', customer.id, self.server_id, {'T1': customer.T1})
        self.system.try_forced_service_on_idle_servers()

    def process_s21(self, customer: 'Customer'):
        self.busy = True
        service_time = random.expovariate(self.system.service_rate_2)
        customer.start_service_2_time = self.env.now
        customer.W2 = self.env.now - customer.T1 - customer.arrival_time
        self.system.record_event('S2_Start', customer.id, self.server_id, {'service_time': service_time})
        yield self.env.timeout(service_time)
        customer.T2 = self.env.now - customer.T1 - customer.arrival_time
        self.system.completed_customers[customer.id] = customer
        self.busy = False
        self.system.record_event('S2_Complete', customer.id, self.server_id, {'T2': customer.T2})
        self.system.try_forced_service_on_idle_servers()

    def process_s2(self, customer: 'Customer'):
        self.busy = True
        service_time = random.expovariate(self.system.service_rate_2)
        customer.start_service_2_time = self.env.now
        customer.W2 = self.env.now - customer.T1 - customer.arrival_time
        self.system.record_event('S2_Start', customer.id, self.server_id, {'service_time': service_time})
        yield self.env.timeout(service_time)
        customer.T2 = self.env.now - customer.T1 - customer.arrival_time
        self.system.completed_customers[customer.id] = customer
        self.busy = False
        self.system.record_event('S2_Complete', customer.id, self.server_id, {'T2': customer.T2})
        self.system.try_forced_service_on_idle_servers()


class Customer:
    def __init__(self, id, arrival_time):
        self.id = id
        self.arrival_time = arrival_time
        self.start_service_1_time = None
        self.start_service_2_time = None
        self.W1 = None
        self.W2 = None
        self.T1 = None
        self.T2 = None
        # NEW: abandonment tracking
        self.patience_remaining: Optional[float] = None
        self.abandon_deadline: Optional[float] = None
