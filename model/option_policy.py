import math
import torch
import torch.nn.functional as F
from utils.model_util import make_module, make_module_list, make_activation
from utils.config import Config

# this policy uses one-step option, the initial option is fixed as o=dim_c


class Policy(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(Policy, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.device = torch.device(config.device)
        self.log_clamp = config.log_clamp_policy
        activation = make_activation(config.activation)
        n_hidden_pi = config.hidden_policy

        self.policy = make_module(self.dim_s, self.dim_a, n_hidden_pi, activation)
        self.a_log_std = torch.nn.Parameter(torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))

        self.to(self.device)

    def a_mean_logstd(self, s):
        y = self.policy(s)
        mean, logstd = y, self.a_log_std.expand_as(y)
        return mean.clamp(-10, 10), logstd.clamp(self.log_clamp[0], self.log_clamp[1])

    def log_prob_action(self, s, a):
        mean, logstd = self.a_mean_logstd(s)
        return (-((a - mean) ** 2) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

    def sample_action(self, s, fixed=False):
        action_mean, action_log_std = self.a_mean_logstd(s)
        if fixed:
            action = action_mean
        else:
            eps = torch.empty_like(action_mean).normal_()
            action = action_mean + action_log_std.exp() * eps
        return action

    def policy_log_prob_entropy(self, s, a):
        mean, logstd = self.a_mean_logstd(s)
        log_prob = (-(a - mean).square() / (2 * (logstd * 2).exp()) - logstd - 0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def get_param(self, low_policy=True):
        if not low_policy:
            print("WARNING >>>> policy do not have high policy params, returning low policy params instead")
        return list(self.parameters())


class OptionPolicy(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(OptionPolicy, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dim_c = config.dim_c
        self.device = torch.device(config.device)
        self.log_clamp = config.log_clamp_policy
        self.is_shared = config.shared_policy
        activation = make_activation(config.activation)
        n_hidden_pi = config.hidden_policy
        n_hidden_opt = config.hidden_option

        if self.is_shared:
            # output prediction p(ct| st, ct-1) with shape (N x ct-1 x ct)
            self.option_policy = make_module(self.dim_s, (self.dim_c+1) * self.dim_c, n_hidden_pi, activation)
            self.policy = make_module(self.dim_s, self.dim_c * self.dim_a, n_hidden_opt, activation)

            self.a_log_std = torch.nn.Parameter(torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))
        else:
            self.policy = make_module_list(self.dim_s, self.dim_a, n_hidden_pi, self.dim_c, activation)
            self.a_log_std = torch.nn.ParameterList([
                torch.nn.Parameter(torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.)) for _ in range(self.dim_c)])
            # i-th model output prediction p(ct|st, ct-1=i)
            self.option_policy = make_module_list(self.dim_s, self.dim_c, n_hidden_opt, self.dim_c+1, activation)

        self.to(self.device)

    def a_mean_logstd(self, st, ct=None):
        # ct: None or long(N x 1)
        # ct: None for all c, return (N x dim_c x dim_a); else return (N x dim_a)
        # s: N x dim_s, c: N x 1, c should always < dim_c
        if self.is_shared:
            mean = self.policy(st).view(-1, self.dim_c, self.dim_a)
            logstd = self.a_log_std.expand_as(mean[:, 0, :])
        else:
            mean = torch.stack([m(st) for m in self.policy], dim=-2)
            logstd = torch.stack([m.expand_as(mean[:, 0, :]) for m in self.a_log_std], dim=-2)
        if ct is not None:
            ind = ct.view(-1, 1, 1).expand(-1, 1, self.dim_a)
            mean = mean.gather(dim=-2, index=ind).squeeze(dim=-2)
            logstd = logstd.gather(dim=-2, index=ind).squeeze(dim=-2)
        return mean.clamp(-10, 10), logstd.clamp(self.log_clamp[0], self.log_clamp[1])

    def switcher(self, s):
        if self.is_shared:
            return self.option_policy(s).view(-1, self.dim_c+1, self.dim_c)
        else:
            return torch.stack([m(s) for m in self.option_policy], dim=-2)

    def get_param(self, low_policy=True):
        if low_policy:
            if self.is_shared:
                return list(self.policy.parameters()) + [self.a_log_std]
            else:
                return list(self.policy.parameters()) + list(self.a_log_std.parameters())
        else:
            return list(self.option_policy.parameters())

    # ===================================================================== #

    def log_trans(self, st, ct_1=None):
        # ct_1: long(N x 1) or None
        # ct_1: None: direct output p(ct|st, ct_1): a (N x ct_1 x ct) array where ct is log-normalized
        unnormed_pcs = self.switcher(st)
        log_pcs = unnormed_pcs.log_softmax(dim=-1)
        if ct_1 is None:
            return log_pcs
        else:
            return log_pcs.gather(dim=-2, index=ct_1.view(-1, 1, 1).expand(-1, 1, self.dim_c)).squeeze(dim=-2)

    def log_prob_action(self, st, ct, at):
        # if c is None, return (N x dim_c x 1), else return (N x 1)
        mean, logstd = self.a_mean_logstd(st, ct)
        if ct is None:
            at = at.view(-1, 1, self.dim_a)
        return (-((at - mean).square()) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

    def log_prob_option(self, st, ct_1, ct):
        log_tr = self.log_trans(st, ct_1)
        return log_tr.gather(dim=-1, index=ct)

    def sample_action(self, st, ct, fixed=False):
        action_mean, action_log_std = self.a_mean_logstd(st, ct)
        if fixed:
            action = action_mean
        else:
            eps = torch.empty_like(action_mean).normal_()
            action = action_mean + action_log_std.exp() * eps
        return action

    def sample_option(self, st, ct_1, fixed=False):
        log_tr = self.log_trans(st, ct_1)
        if fixed:
            return log_tr.argmax(dim=-1, keepdim=True)
        else:
            return F.gumbel_softmax(log_tr, hard=False).multinomial(1).long()

    def policy_entropy(self, st, ct):
        _, log_std = self.a_mean_logstd(st, ct)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
        return entropy.sum(dim=-1, keepdim=True)

    def option_entropy(self, st, ct_1):
        log_tr = self.log_trans(st, ct_1)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return entropy

    def policy_log_prob_entropy(self, st, ct, at):
        mean, logstd = self.a_mean_logstd(st, ct)
        log_prob = (-(at - mean).pow(2) / (2 * (logstd * 2).exp()) - logstd - 0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def option_log_prob_entropy(self, st, ct_1, ct):
        # c1 can be dim_c, c2 should always < dim_c
        log_tr = self.log_trans(st, ct_1)
        log_opt = log_tr.gather(dim=-1, index=ct)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return log_opt, entropy

    def log_alpha_beta(self, s_array, a_array):
        log_pis = self.log_prob_action(s_array, None, a_array).view(-1, self.dim_c)  # demo_len x ct
        log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1 + 1) x ct
        log_tr0 = log_trs[0, -1]
        log_trs = log_trs[1:, :-1]  # (demo_len-1) x ct_1 x ct

        log_alpha = [log_tr0 + log_pis[0]]
        for log_tr, log_pi in zip(log_trs, log_pis[1:]):
            log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) + log_tr).logsumexp(dim=0) + log_pi
            log_alpha.append(log_alpha_t)

        log_beta = [torch.zeros(self.dim_c, dtype=torch.float32, device=self.device)]
        for log_tr, log_pi in zip(reversed(log_trs), reversed(log_pis[1:])):
            log_beta_t = ((log_beta[-1] + log_pi).unsqueeze(dim=0) + log_tr).logsumexp(dim=-1)
            log_beta.append(log_beta_t)
        log_beta.reverse()

        log_alpha = torch.stack(log_alpha)
        log_beta = torch.stack(log_beta)
        entropy = -(log_trs * log_trs.exp()).sum(dim=-1)
        return log_alpha, log_beta, log_trs, log_pis, entropy

    def viterbi_path(self, s_array, a_array):
        #c = torch.zeros(s_array.size(0)+1, 1, dtype=torch.long, device=self.device)
        #c[0] = self.dim_c
        #c[1] = torch.randint(0, self.dim_c, (1, 1))
        #for i in range(2, s_array.size(0)+1):
            #if torch.randint(0, 17, (1,)) == 0:
                #c[i] = torch.randint(0, self.dim_c, (1, 1))
            #else:
                #c[i] = c[i-1]
        #return c, torch.zeros(1, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_pis = self.log_prob_action(s_array, None, a_array).view(-1, 1, self.dim_c)  # demo_len x 1 x ct
            log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1+1) x ct
            log_prob = log_trs[:, :-1] + log_pis
            log_prob0 = log_trs[0, -1] + log_pis[0, 0]
            # forward
            max_path = torch.empty(s_array.size(0), self.dim_c, dtype=torch.long, device=self.device)
            accumulate_logp = log_prob0
            max_path[0] = self.dim_c
            for i in range(1, s_array.size(0)):
                accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
            # backward
            c_array = torch.zeros(s_array.size(0)+1, 1, dtype=torch.long, device=self.device)
            log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
            for i in range(s_array.size(0), 0, -1):
                c_array[i-1] = max_path[i-1][c_array[i]]
        return c_array.detach(), log_prob_traj.detach()


class MoEPolicy(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(MoEPolicy, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dim_c = config.dim_c
        self.policies = torch.nn.ModuleList([Policy(config, dim_s, dim_a) for _ in range(self.dim_c)])
        self.device = torch.device(config.device)

        activation = make_activation(config.activation)
        n_hidden_opt = config.hidden_option

        self.mixer = make_module(self.dim_s, self.dim_c, n_hidden_opt, activation)

        self.to(self.device)

    def get_param(self, low_policy=True):
        if low_policy:
            return list(self.policies.parameters())
        else:
            return list(self.mixer.parameters())

    def mix(self, s):
        # N x C x 1
        return self.mixer(s).softmax(dim=-1).unsqueeze(dim=-1)

    def a_mean_logstd(self, s):
        # N x C x A, N x C x A
        means, logstds = zip(*[p.a_mean_logstd(s) for p in self.policies])
        return torch.stack(means, dim=1), torch.stack(logstds, dim=1)

    # ===================================================================== #

    def policy_log_prob_entropy(self, st, ats):
        # N x C x 1, N x C x 1
        means, logstds = self.a_mean_logstd(st)
        log_probs = (-(ats - means).square() / (2 * (logstds * 2).exp()) - logstds - 0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
        entropies = (0.5 + 0.5 * math.log(2 * math.pi) + logstds).sum(dim=-1, keepdim=True)
        return log_probs, entropies

    def log_prob_action(self, st, ats):
        # N x C x 1
        means, logstds = self.a_mean_logstd(st)
        return (-((ats - means).square()) / (2 * (logstds * 2).exp()) - logstds - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

    def sample_action(self, st, fixed=False):
        raw_action = torch.stack([p.sample_action(st, fixed) for p in self.policies], dim=1)  # N x C x A
        weight = self.mix(st)  # N x C x 1
        action = (raw_action * weight).sum(dim=-2)
        return action, raw_action

