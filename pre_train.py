#!/usr/bin/env python3

import math
import torch
from typing import Union
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.option_policy import OptionPolicy, Policy
from utils.utils import validate, reward_validate
from utils.state_filter import StateFilter


def policy_loss(policy: Policy, sa_array, factor=0.):
    ss_array = torch.cat([s_array for s_array, a_array in sa_array], dim=0)
    as_array = torch.cat([a_array for s_array, a_array in sa_array], dim=0)
    loss = F.mse_loss(policy.sample_action(ss_array, fixed=False), as_array)
    return loss


def policy_loss_option_v2(opolicy: OptionPolicy, sa_array, factor=1.):
    TRLs = []
    RCLs = []
    entropies = []
    for s_array, a_array in sa_array:
        demo_len = s_array.size(0)
        log_alpha, log_beta, log_trs, log_pis, entropy = opolicy.log_alpha_beta(s_array, a_array)

        pc = (log_alpha + log_beta).softmax(dim=-1)
        RCLs.append(-(pc * log_pis.unsqueeze(dim=-2)).sum())

        la_1 = log_alpha[:-1]
        lb_1 = log_beta[1:]
        lpi_1 = log_pis[1:]
        ltr_1 = log_trs[1:]
        pc2 = (la_1.unsqueeze(dim=-1) + (lb_1 + lpi_1).unsqueeze(dim=-2) + ltr_1)
        pc2 = torch.softmax(pc2.view(demo_len-1, -1), dim=-1).view_as(ltr_1)
        TRLs.append(-(pc2 * ltr_1).sum())
    return (sum(RCLs) + sum(TRLs) - factor * sum(entropies)) / len(sa_array)


def policy_loss_option_v3(opolicy: OptionPolicy, sa_array, factor=1.):
    losses = []
    entropies = []
    for s_array, a_array in sa_array:
        log_alpha, log_beta, _, _, entropy = opolicy.log_alpha_beta(s_array, a_array)
        losses.append(-((log_alpha + log_beta).softmax(dim=-1) * (log_alpha + log_beta)).mean())
        entropies.append(entropy.mean())
    return (sum(losses) - factor * sum(entropies)) / len(sa_array)



def pretrain_mini(policy: Union[OptionPolicy, Policy], env, sa_array, logger, msg, in_pretrain=True):
    from envir.circle_env import display_result, display_option_result

    is_option = isinstance(policy, OptionPolicy)
    optimizer = torch.optim.Adam(policy.parameters(), weight_decay=1.e-3)

    log_test = logger.log_pretrain if in_pretrain else logger.log_test
    log_train = logger.log_pretrain if in_pretrain else logger.log_train
    log_test_fig = logger.log_pretrain_fig if in_pretrain else logger.log_test_fig

    epochs = 750
    for i in range(epochs):
        optimizer.zero_grad()
        if is_option:
            loss = policy_loss_option_v3(policy, sa_array, factor=100. * math.exp(-i / 20.))
        else:
            loss = policy_loss(policy, sa_array, factor=100. * math.exp(-i / 20.))
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            v_l, cs = validate(policy, sa_array)
            log_test("expert_logp", v_l, i)
            fig = plt.figure()
            if is_option:
                display_option_result(policy, env)
            else:
                display_result(policy, env)
            log_test_fig("fig", fig, i)
            print(f"pre-{i}: L={loss.item()}, log_p={v_l} ; {msg}")
        else:
            print(f"pre-{i} ; loss={loss.item()} ; {msg}")
        log_train("loss", loss.item(), i)
        logger.flush()


def pretrain(policy: Union[OptionPolicy, Policy], sampler, sa_array, save_name_f, logger, msg, n_iter, log_interval, in_pretrain=True):
    is_option = isinstance(policy, OptionPolicy)
    optimizer = torch.optim.Adam(policy.parameters(), weight_decay=1.e-3)

    log_test = logger.log_pretrain if in_pretrain else logger.log_test
    log_train = logger.log_pretrain if in_pretrain else logger.log_train
    log_test_fig = logger.log_pretrain_fig if in_pretrain else logger.log_test_fig
    log_test_info = logger.log_pretrain_info if in_pretrain else logger.log_test_info

    sa_array = sampler.filter_demo(sa_array)

    for i in range(n_iter):
        optimizer.zero_grad()
        if is_option:
            loss = policy_loss_option_v3(policy, sa_array, factor=500. * math.exp(-i / 200.))
        else:
            loss = policy_loss(policy, sa_array, factor=500. * math.exp(-i / 200.))
        loss.backward()
        optimizer.step()

        if (i + 1) % log_interval == 0:
            v_l, cs_expert = validate(policy, sa_array)
            log_test("expert_logp", v_l, i)
            info_dict, cs_sample = reward_validate(sampler, policy, do_print=True)

            if is_option:
                a = plt.figure()
                a.gca().plot(cs_expert[0])
                log_test_fig("expert_c", a, i)
                a = plt.figure()
                a.gca().plot(cs_sample[0])
                log_test_fig("sample_c", a, i)
            log_test_info(info_dict, i)

            torch.save((policy.state_dict(), StateFilter().state_dict()), save_name_f(i))
            print(f"pre-{i} ; loss={loss.item()} ; log_p={v_l} ; {msg}")
        else:
            print(f"pre-{i} ; loss={loss.item()} ; {msg}")
        log_train("loss", loss.item(), i)
        logger.flush()
