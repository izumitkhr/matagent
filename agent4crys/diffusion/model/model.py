import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .util import lattice_params_to_matrix_torch
from .decoder import get_decoder
from ..embedding import get_time_embedding
from ..scheduler import get_scheduler
from ..transition import get_transition
from ..util.loss import get_loss_func
from ..transition import (
    CategoricalTransition,
    ContinuousTransition,
    GeneralCategoricalTransition,
)


class CrystalDiffusion(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.cfg = cfg_model
        self.num_classes = 100

        # decoder
        self.decoder = get_decoder(cfg_model)

        # time embedding
        self.time_dim = self.cfg.time_embedding.dim  # 256
        self.time_embedding = get_time_embedding(cfg_model)

        # scheduler
        self.timesteps = self.cfg.timesteps
        self.lat_scheduler = get_scheduler(self.cfg.scheduler.lat, self.timesteps)
        self.coord_scheduler = get_scheduler(self.cfg.scheduler.coord, self.timesteps)
        # self.type_scheduler = get_scheduler(self.cfg.scheduler.type, self.timesteps)

        # transition
        self.lat_ts = get_transition(self.cfg.scheduler.lat, self.lat_scheduler)
        self.coord_ts = get_transition(self.cfg.scheduler.coord, self.coord_scheduler)
        # self.type_ts = get_transition(self.cfg.scheduler.type, self.type_scheduler)
        # self.type_ts_mode = self._get_transition_mode()

        # loss
        self.loss_func = get_loss_func(self.cfg)

    def _get_transition_mode(self):
        if isinstance(
            self.type_ts, (CategoricalTransition, GeneralCategoricalTransition)
        ):
            return "cat"
        elif isinstance(self.type_ts, ContinuousTransition):
            return "cont"
        else:
            raise Exception(f"Unsupported transition mode: {type(self.type_ts)}")

    def forward(self, batch):
        batch_size = batch.num_graphs

        # sample timesteps
        times = self.lat_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        # gt data M(L, F, A)
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords  # [num_nodes, 3]
        atom_types = batch.atom_types - 1  # [num_nodes]

        # perturbed data M'(L',F',A')
        input_l, rand_l = self.lat_ts.add_noise(lattices, times)
        input_f, rand_f, tar_f = self.coord_ts.add_noise(
            frac_coords, times, batch.num_atoms
        )
        # if self.type_ts_mode == "cat":
        #     input_type, log_type_vt, log_type_v0 = self.type_ts.add_noise(
        #         atom_types, times, batch.batch
        #     )
        # if self.type_ts_mode == "cont":
        #     input_type, rand_t = self.type_ts.add_noise(
        #         atom_types, times, batch.num_atoms
        #     )
        input_type = F.one_hot(atom_types, self.num_classes).float()

        # pred
        pred_l, pred_f, pred_t = self.decoder(
            time_emb, input_type, input_f, input_l, batch.num_atoms, batch.batch
        )

        # compute loss
        loss_l = self.lat_ts.get_mse_loss(pred_l, rand_l)
        loss_f = self.coord_ts.get_mse_loss(pred_f, tar_f)
        # if self.type_ts_mode == "cat":
        #     loss_t = self.type_ts.get_loss(
        #         pred_t, log_type_vt, log_type_v0, times, batch.batch
        #     )
        # if self.type_ts_mode == "cont":
        #     loss_t = self.type_ts.get_mse_loss(pred_t, rand_t)
        loss_t = 0.0

        loss = self.loss_func(loss_l, loss_f, loss_t)

        return {
            "loss": loss,
            "loss_lattice": loss_l,
            "loss_coord": loss_f,
            "loss_type": loss_t,
        }

    def compute_stats(self, outputs, prefix):
        loss = outputs["loss"]
        loss_lattice = outputs["loss_lattice"]
        loss_coord = outputs["loss_coord"]
        loss_type = outputs["loss_type"]

        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_loss_lattice": loss_lattice,
            f"{prefix}_loss_coord": loss_coord,
            # f"{prefix}_loss_type": loss_type,
            # f"{prefix}_cost_lattice": self.loss_func.get_current_costs()[
            #     "cost_lattice"
            # ],
            # f"{prefix}_cost_coord": self.loss_func.get_current_costs()["cost_coord"],
            # f"{prefix}_cost_type": self.loss_func.get_current_costs()["cost_type"],
        }

        return log_dict, loss

    @torch.no_grad()
    def sample(self, batch, step_lr=1e-5):
        batch_size = batch.num_graphs
        # initialize
        l_T = self.lat_ts.sample_init((batch_size, 3, 3))
        f_T = self.coord_ts.sample_init((batch.num_nodes, 3))
        # if self.type_ts_mode == "cat":
        #     t_T, t_T_onehot, log_type_vt = self.type_ts.sample_init(batch.num_nodes)
        # if self.type_ts_mode == "cont":
        #     t_T_onehot = self.type_ts.sample_init((batch.num_nodes, self.num_classes))
        #     t_T = t_T_onehot.argmax(dim=-1)
        t_T = batch.atom_types - 1
        t_T_onehot = F.one_hot(t_T, self.num_classes).float()

        traj = {
            self.timesteps: {
                "num_atoms": batch.num_atoms,
                "atom_feat": t_T_onehot,
                "atom_types": t_T + 1,
                "frac_coords": f_T % 1.0,
                "lattices": l_T,
            }
        }

        for t in tqdm(range(self.timesteps, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)

            f_t = traj[t]["frac_coords"]
            l_t = traj[t]["lattices"]
            t_t = traj[t]["atom_feat"]

            # Corrector
            pred_l, pred_f, pred_t = self.decoder(
                time_emb, t_t, f_t, l_t, batch.num_atoms, batch.batch
            )
            l_t_minus_05 = l_t
            f_t_minus_05 = self.coord_ts.correct(f_t, pred_f, t, step_lr)
            t_t_minus_05 = t_t
            f_t_minus_05 = f_t_minus_05 % 1.0

            # Predictor
            pred_l, pred_f, pred_t = self.decoder(
                time_emb,
                t_t_minus_05,
                f_t_minus_05,
                l_t_minus_05,
                batch.num_atoms,
                batch.batch,
            )
            l_t_minus_1 = self.lat_ts.predict(l_t_minus_05, pred_l, t)
            f_t_minus_1 = self.coord_ts.predict(f_t_minus_05, pred_f, t)
            # if self.type_ts_mode == "cat":
            #     log_type_vt, t_t_minus_1, t_t_minus_1_onehot = self.type_ts.predict(
            #         log_type_vt, pred_t, times, batch.batch
            #     )
            # else:
            #     t_t_minus_1_onehot = self.type_ts.predict(t_t_minus_05, pred_t, t)
            #     t_t_minus_1 = t_t_minus_1_onehot.argmax(dim=-1)

            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_feat": t_T_onehot,
                "atom_types": t_T + 1,
                "frac_coords": f_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
            }

        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": torch.stack(
                [traj[i]["atom_types"] for i in range(self.timesteps, -1, -1)]
            ),
            "all_frac_coords": torch.stack(
                [traj[i]["frac_coords"] for i in range(self.timesteps, -1, -1)]
            ),
            "all_lattices": torch.stack(
                [traj[i]["lattices"] for i in range(self.timesteps, -1, -1)]
            ),
        }

        return traj[0], traj_stack
