import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter

from ...embedding.dist import SinusoidsEmbedding


class CSPLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True,
    ):
        super().__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        assert dis_emb is not None
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim

        # edge mlp
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        # node mlp
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], edge_index[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        # lattice feat
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flattedn_edges = lattice_ips_flatten[edge2graph]
        # edge feat (FT)
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        # concat feat
        edges_input = torch.cat([hi, hj, lattice_ips_flattedn_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(
            edge_features,
            edge_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,  # TODO: checnge to temb ?
        num_layers=4,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        num_classes=100,
        ln=False,
        ip=True,
    ):
        super().__init__()
        self.ip = ip
        self.num_layers = num_layers
        self.edge_style = edge_style
        self.node_embedding = nn.Linear(num_classes, hidden_dim)
        self.node_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise NotImplementedError(act_fn)

        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        else:
            raise NotImplementedError(dis_emb)

        for i in range(num_layers):
            self.add_module(
                "csp_layer%d" % i,
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )

        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # for output
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)
        self.type_out = nn.Linear(hidden_dim, num_classes)

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])
        else:
            raise NotImplementedError(f"edge_style={self.edge_style}")

    def forward(self, temb, atom_types, frac_coords, lattices, num_atoms, node2graph):
        # gen edges
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        # gen node feat
        node_features = self.node_embedding(atom_types)
        temb_per_atom = temb.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, temb_per_atom], dim=1)
        node_features = self.node_latent_emb(node_features)
        # update node features
        for i in range(self.num_layers):
            node_features = self._modules["csp_layer%d" % i](
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )
        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # final output
        # frac coords
        out_f = self.coord_out(node_features)
        # atom types
        # out_t = self.type_out(node_features)
        # lattice
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        out_l = self.lattice_out(graph_features)
        out_l = out_l.view(-1, 3, 3)
        if self.ip:
            out_l = torch.einsum("bij,bjk->bik", out_l, lattices)
        # return out_l, out_f, out_t
        return out_l, out_f, None
