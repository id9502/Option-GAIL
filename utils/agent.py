import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.option_policy import OptionPolicy, MoEPolicy
from utils.state_filter import StateFilter
from utils.utils import set_seed


__all__ = ["Sampler"]

# rlbench: 4096, 1t, 135s; 2t, 79s; 4t, 51s; 6t, 51s
# mujoco: 5000, 1t, 7.2s; 2t, 5.6s; 4t, 4.2s; 6t, 4.2s


class _sQueue(object):
    def __init__(self, pipe_rw, r_lock, w_lock):
        self.rlock = r_lock
        self.wlock = w_lock
        self.pipe_rw = pipe_rw

    def __del__(self):
        self.pipe_rw.close()

    def get(self, time_out=0.):
        d = None
        if self.pipe_rw.poll(time_out):
            with self.rlock:
                d = self.pipe_rw.recv()
        return d

    def send(self, d):
        with self.wlock:
            self.pipe_rw.send(d)


def pipe_pair():
    p_lock = Lock()
    c_lock = Lock()
    pipe_c, pipe_p = Pipe(duplex=True)
    child_q = _sQueue(pipe_c, p_lock, c_lock)
    parent_q = _sQueue(pipe_p, c_lock, p_lock)
    return child_q, parent_q


def option_loop(env, policy, state_filter, fixed):
    with torch.no_grad():
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        c_array.append(ct)
        while not done:
            st = torch.as_tensor(state_filter(s, fixed), dtype=torch.float32, device=policy.device).unsqueeze(0)
            ct = policy.sample_option(st, ct, fixed=fixed).detach()
            at = policy.sample_action(st, ct, fixed=fixed).detach()
            s_array.append(st)
            c_array.append(ct)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        c_array = torch.cat(c_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, c_array, a_array, r_array


def loop(env, policy, state_filter, fixed):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        while not done:
            st = torch.as_tensor(state_filter(s, fixed), dtype=torch.float32, device=policy.device).unsqueeze(0)
            at = policy.sample_action(st, fixed=fixed).detach()
            s_array.append(st)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, a_array, r_array


def moe_loop(env, policy: MoEPolicy, state_filter, fixed):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        while not done:
            st = torch.as_tensor(state_filter(s, fixed), dtype=torch.float32, device=policy.device).unsqueeze(0)
            at, raw_at = policy.sample_action(st, fixed=fixed)
            s_array.append(st)
            a_array.append(raw_at.detach())
            s, r, done = env.step(at.detach().cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, a_array, r_array


class _SamplerCommon(object):
    def __init__(self, seed, policy, use_state_filter):
        self.state_filter = StateFilter(enable=use_state_filter)
        self.device = policy.device

    def collect(self, policy_param, n_sample, fixed=False):
        raise NotImplementedError()

    def filter_demo(self, sa_array):
        if self.state_filter.enable:
            return tuple((self.state_filter(s, fixed=True).to(self.device), a) for s, a in sa_array)
        else:
            return sa_array

    def state_dict(self):
        return self.state_filter.state_dict()

    def load_state_dict(self, running_state_dict):
        self.state_filter.load_state_dict(running_state_dict)


class _Sampler(_SamplerCommon):
    def __init__(self, seed, env, policy, use_state_filter, n_thread=4, loop_func=None):
        super(_Sampler, self).__init__(seed, policy, use_state_filter=use_state_filter)
        self.counter = Value('i', 0)
        self.state = Value('i', n_thread)
        child_q, self.queue = pipe_pair()
        self.procs = [Process(target=self.worker, name=f"subproc_{seed}",
                              args=(seed, env, policy, self.state_filter, loop_func, self.state, self.counter, child_q))
                      for _ in range(n_thread)]
        self.pids = []
        for p in self.procs:
            p.daemon = True
            p.start()
            self.pids.append(p.pid)

        while self.state.value > 0:
            time.sleep(0.1)

    def collect(self, policy_param, n_sample, fixed=False):
        # n_sample <0 for number of trajectories, >0 for number of sa pairs
        for _ in self.procs:
            self.queue.send((policy_param, self.state_filter.state_dict(), fixed))

        with self.state.get_lock():
            self.state.value = -len(self.procs)

        while self.state.value < 0:
            time.sleep(0.1)

        with self.counter.get_lock():
            self.counter.value = n_sample

        with self.state.get_lock():
            self.state.value = len(self.procs)

        ret = []
        flt = None
        while self.state.value > 0:
            d = self.queue.get(0.0001)
            while d is not None:
                traj, filter_param = d
                ret.append(tuple(x.to(self.device) for x in traj))
                flt = filter_param
                d = self.queue.get(0.0001)
        self.state_filter.load_state_dict(flt)
        return ret

    def __del__(self):
        print(f"agent process is terminated, check if any subproc left: {self.pids}")
        for p in self.procs:
            p.terminate()

    @staticmethod
    def worker(seed: int, env, policy, state_filter, loop_func, state: Value, counter: Value, queue: _sQueue):
        # state 0: idle, -n: init param, n: sampling
        set_seed(seed)

        env.init(display=False)
        with state.get_lock():
            state.value -= 1

        while True:
            while state.value >= 0:
                time.sleep(0.1)

            d = None
            while d is None:
                d = queue.get(5)

            net_param, filter_param, fixed = d
            policy.load_state_dict(net_param)
            state_filter.load_state_dict(filter_param)

            with state.get_lock():
                state.value += 1

            while state.value <= 0:
                time.sleep(0.1)

            while state.value > 0:
                traj = loop_func(env, policy, state_filter, fixed=fixed)
                with counter.get_lock():
                    if counter.value > 0:
                        queue.send((tuple(x.cpu() for x in traj), state_filter.state_dict()))
                        counter.value -= traj[0].size(0)
                        if counter.value <= 0:
                            counter.value = 0
                            with state.get_lock():
                                state.value = 0
                    elif counter.value < 0:
                        queue.send((tuple(x.cpu() for x in traj), state_filter.state_dict()))
                        counter.value += 1
                        if counter.value >= 0:
                            counter.value = 0
                            with state.get_lock():
                                state.value = 0


class _SamplerSS(_SamplerCommon):
    def __init__(self, seed, env, policy, use_state_filter, n_thread=1, loop_func=None):
        super(_SamplerSS, self).__init__(seed, policy, use_state_filter=use_state_filter)
        if n_thread > 1:
            print(f"Warning: you are using single thread sampler, despite n_thread={n_thread}")
        self.env = deepcopy(env)
        self.env.init(display=False)
        self.policy = deepcopy(policy)
        self.loop_func = loop_func

    def collect(self, policy_param, n_sample, fixed=False):
        self.policy.load_state_dict(policy_param)
        counter = n_sample
        rets = []
        if counter > 0:
            while counter > 0:
                traj = self.loop_func(self.env, self.policy, self.state_filter, fixed=fixed)
                rets.append(traj)
                counter -= traj[0].size(0)
        else:
            while counter < 0:
                traj = self.loop_func(self.env, self.policy, self.state_filter, fixed=fixed)
                rets.append(traj)
                counter += 1
        return rets


def Sampler(seed, env, policy, use_state_filter: bool = True, n_thread=4) -> _SamplerCommon:
    if isinstance(policy, OptionPolicy):
        loop_func = option_loop
    elif isinstance(policy, MoEPolicy):
        loop_func = moe_loop
    else:
        loop_func = loop
    class_m = _Sampler if n_thread > 1 else _SamplerSS
    return class_m(seed, env, policy, use_state_filter, n_thread, loop_func)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    set_start_method("spawn")
