import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f"Error: {x.max().item >= {num_classes}}"
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def unsqueeze(tensor, ndim=2):
    if ndim == 1:
        return tensor
    elif ndim == 2:
        return tensor.unsqueeze(-1)
    elif ndim == 3:
        return tensor.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError("ndim > 3")


def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    return unsqueeze(out, ndim)
    # if ndim == 1:
    #     return out
    # elif ndim == 2:
    #     return out.unsqueeze(-1)
    # elif ndim == 3:
    #     return out.unsqueeze(-1).unsqueeze(-1)
    # else:
    #     raise NotImplementedError("ndim > 3")


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    # gumbel dist.
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl


def log_categorical(log_x_start, log_prob):
    # for log p(x_0 | x_1)
    return (log_x_start.exp() * log_prob).sum(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


class CategoricalTransition(nn.Module):
    def __init__(self, scheduler, num_classes):
        super().__init__()
        self.scheduler = scheduler
        self.num_classes = torch.tensor(num_classes)

    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    def add_noise(self, v, timestep, batch):
        # v_t = a * v_0 + (1-a) / K
        log_node_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_node_vt = self.q_vt_sample(log_node_v0, timestep, batch)
        v_perturbed = F.one_hot(v_perturbed, self.num_classes).float()
        return v_perturbed, log_node_vt, log_node_v0

    def q_vt_sample(self, log_v0, timestep, batch):
        # Sample from q(v_t | v_0)
        log_q_vt_v0 = self.q_vt_pred(log_v0, timestep, batch)
        sample_class = log_sample_categorical(log_q_vt_v0)
        log_sample = index_to_log_onehot(sample_class, self.num_classes)
        return sample_class, log_sample

    def q_vt_pred(self, log_v0, timestep, batch):
        # Compute q(v_t | v_0)
        ndim = log_v0.ndim
        log_alpha_bar_t = extract(
            self.scheduler.log_cumprod_alphas, timestep, batch, ndim=ndim
        )
        log_1_min_alpha_bar = extract(
            self.scheduler.log_1_min_cumprod_alphas, timestep, batch, ndim=ndim
        )

        log_probs = log_add_exp(
            log_v0 + log_alpha_bar_t, log_1_min_alpha_bar - torch.log(self.num_classes)
        )
        return log_probs

    def q_v_pred_one_timestep(self, log_vt_1, timestep, batch):
        # q(v_t | v_t-1)
        ndim = log_vt_1.ndim
        log_alpha_t = extract(self.scheduler.log_alphas, timestep, batch, ndim=ndim)
        log_1_min_alpha_t = extract(
            self.scheduler.log_1_min_alphas, timestep, batch, ndim=ndim
        )
        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t, log_1_min_alpha_t - torch.log(self.num_classes)
        )
        return log_probs

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(v_t-1 | v_t, v_0) = q(v_t | v_t-1) * q(v_t-1 | v_0) / q(v_t | v_0)
        t_minus_1 = t - 1
        t_minus_1 = torch.where(
            t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1
        )  # Remove negative values, will not be used anyway for final decoder

        # q(v_t-1 | v_0)
        log_qvtmin_v0 = self.q_vt_pred(log_v0, t_minus_1, batch)

        ndim = log_v0.ndim
        if ndim == 2:
            t_expand = t[batch].unsqueeze(-1)
        else:
            raise NotImplementedError(f"ndim={ndim}")
        log_qvtmin_v0 = torch.where(t_expand == 0, log_v0, log_qvtmin_v0)

        unnormed_logprobs = log_qvtmin_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vtmin_given_vt_v0 = unnormed_logprobs - torch.logsumexp(
            unnormed_logprobs, dim=-1, keepdim=True
        )
        return log_vtmin_given_vt_v0

    def compute_v_Lt(self, log_v_post_true, log_v_post_pred, log_v0, t, batch):
        kl_v = categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = -log_categorical(log_v0, log_v_post_pred)

        ndim = log_v_post_true.ndim
        if ndim == 2:
            mask = (t == 0).float()[batch]
        else:
            raise NotImplementedError(f"ndim: {ndim}")
        loss_v = mask * decoder_nll_v + (1 - mask) * kl_v
        return loss_v

    def sample_init(self, n):
        init_log_atom_vt = torch.zeros(n, self.num_classes).to(
            self.scheduler.alphas.device
        )
        # TODO: check ! Now, initializing with uniform distribution.
        init_types = log_sample_categorical(init_log_atom_vt)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt

    def get_loss(self, pred_node_v, log_node_vt, log_node_v0, timestep, batch):
        log_v_recon = F.log_softmax(pred_node_v, dim=-1)
        log_v_post_true = self.q_v_posterior(log_node_v0, log_node_vt, timestep, batch)
        log_v_post_pred = self.q_v_posterior(log_v_recon, log_node_vt, timestep, batch)
        kl_type = self.compute_v_Lt(
            log_v_post_true, log_v_post_pred, log_node_v0, timestep, batch
        )
        return kl_type.mean()

    def predict(self, log_node_vt, v_pred, timestep, batch):
        log_v_recon = F.log_softmax(v_pred, dim=-1)
        log_node_vt_prev = self.q_v_posterior(log_v_recon, log_node_vt, timestep, batch)
        node_type_prev = log_sample_categorical(log_node_vt_prev)
        node_onehot_prev = self.onehot_encode(node_type_prev)
        # TODO: check !
        log_node_vt_prev = index_to_log_onehot(node_type_prev, self.num_classes)
        return log_node_vt_prev, node_type_prev, node_onehot_prev


class GeneralCategoricalTransition(nn.Module):
    def __init__(self, scheduler, num_classes, init_prob):
        super().__init__()
        self.eps = 1e-30
        self.num_classes = num_classes
        self.num_timesteps = len(scheduler.betas)
        self.scheduler = scheduler
        self._set_init_prob(init_prob, num_classes)
        self._construct_transition_mat()

    def _set_init_prob(self, init_prob, num_classes):
        if init_prob is None or init_prob == "uniform":
            # default uniform
            self.init_prob = torch.ones(num_classes) / num_classes
        elif init_prob == "absorb":
            init_prob = 0.001 * torch.ones(num_classes)
            init_prob[0] = 1.0
            self.init_prob = init_prob / torch.sum(init_prob)
        elif init_prob == "tomask":
            init_prob = 0.001 * torch.ones(num_classes)
            init_prob[-1] = 1.0
            self.init_prob = init_prob / torch.sum(init_prob)
        else:
            raise NotImplementedError(f"init_prob: {init_prob}")

    def _construct_transition_mat(self):
        """Compute transition matrix for q(x_t | x_0)"""
        # Construct transition matrices for q(x_t | x_t-1)
        q_onestep_mats = [
            self._get_transition_mat(t) for t in range(0, self.num_timesteps)
        ]
        q_onestep_mats = torch.stack(q_onestep_mats, axis=0)  # (T, K, K)
        transpose_q_onestep_mats = torch.transpose(q_onestep_mats, 2, 1)
        self.transpoes_q_onestep_mats = nn.Parameter(
            transpose_q_onestep_mats, requires_grad=False
        )

        # Construct transition matrices for q(x_t | x_0)
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = q_mat_t @ q_onestep_mats[t]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, axis=0)
        self.q_mats = nn.Parameter(q_mats, requires_grad=False)

    def _get_transition_mat(self, timestep):
        """Computes transition matrix for q(x_t | x_t-1)

        Args:
            timestep (integer): timestep

        Returns:
            Q_t: transition matrix.
        """
        assert self.init_prob is not None
        beta_t = self.scheduler.betas[timestep]
        mat = self.init_prob.unsqueeze(0)
        mat = mat.repeat(self.num_classes, 1)
        mat = beta_t * mat
        mat_diag = torch.eye(self.num_classes) * (1.0 - beta_t)
        mat = mat + mat_diag
        return mat

    def add_noise(self, v, timestep, batch):
        log_node_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_node_vt = self.q_vt_sample(log_node_v0, timestep, batch)
        v_perturbed = F.one_hot(v_perturbed, self.num_classes).float()
        return v_perturbed, log_node_vt, log_node_v0

    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    def q_vt_sample(self, log_v0, timestep, bacth):
        # Sample form q(v_t | v_0)
        log_q_vt_v0 = self.q_vt_pred(log_v0, timestep, bacth)
        sample_class = log_sample_categorical(log_q_vt_v0)
        log_sample = index_to_log_onehot(sample_class, self.num_classes)
        return sample_class, log_sample

    def q_vt_pred(self, log_v0, timestep, batch):
        # Compute q(v_t | v_0)
        qt_mat = extract(self.q_mats, timestep, batch, ndim=1)
        q_vt = torch.einsum("...i,...ij->...j", log_v0.exp(), qt_mat)
        return torch.log(q_vt + self.eps).clamp(-32.0)

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(v_t-1 | v_t, v_0) = q(v_t | v_t-1) * q(v_t-1 | v_0) / q(v_t | v_0)
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)

        # x_t (Q_t)^T
        fact1 = extract(self.transpoes_q_onestep_mats, t, batch, ndim=1)
        fact1 = torch.einsum("bj,bjk->bk", torch.exp(log_vt), fact1)

        # x_0 Q_t-1
        fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)  # (batch, N, N)
        fact2 = torch.einsum("bj,bjk->bk", torch.exp(log_v0), fact2)  # (batch, N)

        fact1 = torch.log(fact1 + self.eps).clamp_min(-32.0)
        fact2 = torch.log(fact2 + self.eps).clamp_min(-32.0)
        out = fact1 + fact2
        out = out - torch.logsumexp(out, dim=-1, keepdim=True)

        ndim = log_v0.ndim
        t_expand = unsqueeze(t[batch], ndim=ndim)
        out_t0 = log_v0
        out = torch.where(t_expand == 0, out_t0, out)
        return out

    def compute_v_Lt(self, log_v_post_true, log_v_post_pred, log_v0, t, batch):
        kl_v = categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = -log_categorical(log_v0, log_v_post_pred)

        ndim = log_v_post_true.ndim
        if ndim == 2:
            mask = (t == 0).float()[batch]
        else:
            raise NotImplementedError(f"ndim: {ndim}")
        loss_v = mask * decoder_nll_v + (1 - mask) * kl_v
        return loss_v

    def sample_init(self, n):
        init_log_atom_vt = (
            torch.log(self.init_prob + self.eps).clamp_min(-32.0).to(self.q_mats.device)
        )
        init_log_atom_vt = init_log_atom_vt.unsqueeze(0).repeat(n, 1)
        init_types = log_sample_categorical(init_log_atom_vt)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt

    def get_loss(self, pred_node_v, log_node_vt, log_node_v0, timestep, batch):
        log_v_recon = F.log_softmax(pred_node_v, dim=-1)
        log_v_post_true = self.q_v_posterior(log_node_v0, log_node_vt, timestep, batch)
        log_v_post_pred = self.q_v_posterior(log_v_recon, log_node_vt, timestep, batch)
        kl_type = self.compute_v_Lt(
            log_v_post_true, log_v_post_pred, log_node_v0, timestep, batch
        )
        return kl_type.mean()

    def predict(self, log_node_vt, v_pred, timestep, batch):
        log_v_recon = F.log_softmax(v_pred, dim=-1)
        log_node_vt_prev = self.q_v_posterior(log_v_recon, log_node_vt, timestep, batch)
        node_type_prev = log_sample_categorical(log_node_vt_prev)
        node_onehot_prev = self.onehot_encode(node_type_prev)
        # TODO: check !
        log_node_vt_prev = index_to_log_onehot(node_type_prev, self.num_classes)
        return log_node_vt_prev, node_type_prev, node_onehot_prev
