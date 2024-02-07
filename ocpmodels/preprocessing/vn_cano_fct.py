import torch
import torch.nn as nn
import torch_scatter as ts
import pytorch_lightning as pl
from ocpmodels.preprocessing.vn_layers.vn_layers import VNSoftplus, VNLeakyReLU
from ocpmodels.preprocessing.vn_layers.set_base_models import SequentialMultiple


class VNShallowNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, nonlinearity='leakyrelu', num_layers=2, dropout=0.0):
        super().__init__()
        self.layer_pooling = "sum"
        self.final_pooling = "sum"
        self.num_layers = num_layers 
        self.nonlinearity = nonlinearity
        self.dropout = dropout

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        self.layer_1 = VNDeepSetLayer(
            self.in_dim, self.hidden_dim, self.nonlinearity
        )
        self.layer_2 = VNDeepSetLayer(
            self.hidden_dim, self.hidden_dim, self.nonlinearity
        )
        self.output_layer = (
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.batch_size = 1
        self.canon_translation = True

    def forward(self, loc, edges):
        # edges : [2, 93]
        nb_atoms = loc.shape[0]

        batch_indices = torch.arange(self.batch_size).reshape(-1, 1)
        batch_indices = batch_indices.repeat(1, nb_atoms).reshape(-1) # [31]

        mean_loc = ts.scatter(loc, batch_indices, 0, reduce=self.layer_pooling)
        mean_loc = mean_loc.repeat(nb_atoms, 1, 1).transpose(0, 1).reshape(-1, 3)
        mean_loc = torch.mean(loc, dim=0)

        canonical_loc = loc - mean_loc # [31, 3]
        features = torch.stack([canonical_loc], dim=2) # [31, 3, 1]

        x, _ = self.layer_1(features, edges)
        x, _ = self.layer_2(x, edges)

        if x.shape[0] > 1:
            x = ts.scatter(x, batch_indices, 0, reduce=self.final_pooling)
        output = self.output_layer(x)
        
        output = output.reshape(-1, 3, 4)
        # breakpoint()
        
        rotation_vectors = output[:, :, :3]
        translation_vectors = output[:, :, 3:] if self.canon_translation else 0.0
        translation_vectors = translation_vectors + mean_loc[None, :, None]

        return rotation_vectors, translation_vectors.squeeze()

# class VNShallowNet(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim=64, nonlinearity='leakyrelu', num_layers=2, dropout=0.0):
#         self.layer_pooling = "sum"
#         self.final_pooling = "sum"
#         self.num_layers = num_layers 
#         self.nonlinearity = nonlinearity
#         self.dropout = dropout

#         self.out_dim = out_dim
#         self.hidden_dim = hidden_dim
#         self.in_dim = in_dim

#         self.first_set_layer = VNDeepSetLayer(
#             self.in_dim, self.hidden_dim, self.nonlinearity, self.layer_pooling, False, dropout=self.dropout
#         )
#         self.set_layers = SequentialMultiple(
#             *[
#                 VNDeepSetLayer(
#                     self.hidden_dim, self.hidden_dim, self.nonlinearity, self.layer_pooling, dropout=self.dropout
#                 )
#                 for i in range(self.num_layers - 1)
#             ]
#         )
#         self.output_layer = (
#             nn.Linear(self.hidden_dim, self.out_dim)
#         )
#         self.batch_size = 1

#     def forward(self, loc, edges):
#         nb_particles = loc.shape[0]

#         batch_indices = torch.arange(self.batch_size, device=self.device).reshape(-1, 1)
#         batch_indices = batch_indices.repeat(1, nb_particles).reshape(-1)

#         mean_loc = ts.scatter(loc, batch_indices, 0, reduce=self.layer_pooling)
#         mean_loc = mean_loc.repeat(nb_particles, 1, 1).transpose(0, 1).reshape(-1, 3)
#         canonical_loc = loc - mean_loc
#         features = torch.stack([canonical_loc], dim=2) 

#         x, _ = self.first_set_layer(features, edges)
#         x, _ = self.set_layers(x, edges)
        
#         x = ts.scatter(x, batch_indices, 0, reduce=self.final_pooling)
#         output = self.output_layer(x)

#         output = output.repeat(nb_particles, 1, 1, 1).transpose(0, 1)
#         output = output.reshape(-1, 3, 4)

#         rotation_vectors = output[:, :, :3]
#         translation_vectors = output[:, :, 3:] if self.canon_translation else 0.0
#         translation_vectors = translation_vectors + mean_loc[:, :, None]
# 
#         return rotation_vectors, translation_vectors.squeeze()
    

class VNDeepSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity, pooling="sum", residual=True, dropout=0.0):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.pooling = pooling
        self.residual = residual
        self.nonlinearity = nonlinearity
        self.dropout = dropout

        self.identity_linear = nn.Linear(in_channels, out_channels)
        self.pooling_linear = nn.Linear(in_channels, out_channels)

        self.dropout_layer = nn.Dropout(self.dropout)

        if self.nonlinearity == "softplus":
            self.nonlinear_function = VNSoftplus(out_channels, share_nonlinearity=False)
        elif self.nonlinearity == "relu":
            self.nonlinear_function = VNLeakyReLU(out_channels, share_nonlinearity=False, negative_slope=0.0)
        elif self.nonlinearity == "leakyrelu":
            self.nonlinear_function = VNLeakyReLU(out_channels, share_nonlinearity=False)

    def forward(self, x, edges):
        # x : [31, 3, X] / [31, 3]
        # edges : [2, 93]
        edges_1 = edges[0]
        edges_2 = edges[1]

        identity = self.identity_linear(x) # [31, 3, 64] / [31, 64]

        nodes_1 = torch.index_select(x, 0, edges_1) # [93, 3, 1]
        pooled_set = ts.scatter(nodes_1, edges_2, 0, reduce=self.pooling) # [31, 3, 1]
        pooling = self.pooling_linear(pooled_set) # [31, 3, 64]

        embedding = identity + pooling # [31, 3, 64]
        embedding = embedding.transpose(1, -1) # [31, 64, 3]
        
        output = self.nonlinear_function(embedding).transpose(1, -1) # [31, 3, 64]
        output = self.dropout_layer(output)

        if self.residual:
            output = output + x

        return output, edges



class BaseEuclideangraphModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        self.weight_decay = hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        self.patience = hyperparams.patience if hasattr(hyperparams, "patience") else 100
        self.edges = [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3],
        ]

        self.loss = nn.MSELoss()

        self.dummy_nodes = torch.zeros(2, 1, device=self.device, dtype=torch.float)
        self.dummy_loc = torch.zeros(2, 3, device=self.device, dtype=torch.float)
        self.dummy_edges = [
            torch.zeros(40, device=self.device, dtype=torch.long),
            torch.zeros(40, device=self.device, dtype=torch.long),
        ]
        self.dummy_vel = torch.zeros(2, 3, device=self.device, dtype=torch.float)
        self.dummy_edge_attr = torch.zeros(40, 2, device=self.device, dtype=torch.float)

    def training_step(self, batch, batch_idx):
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(batch_size, n_nodes)

        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties

        outputs = self(nodes, loc.detach(), edges, vel, edge_attr, charges)

        loss = self.loss(outputs, loc_end)

        # metrics = {"train/loss": loss}
        # self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(batch_size, n_nodes)

        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties

        outputs = self(nodes, loc.detach(), edges, vel, edge_attr, charges)

        loss = self.loss(outputs, loc_end)
        # if self.global_step == 0:
        #     wandb.define_metric("valid/loss", summary="min")

        # metrics = {"valid/loss": loss}
        # self.log_dict(metrics, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-12)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=0.5, min_lr=1e-6, mode="max"
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/loss"}

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        # if self.current_epoch == 0:
        #     model_filename = f"canonical_network/results/nbody/onnx_models/{self.model}_{wandb.run.name}_{str(self.global_step)}.onnx"
        #     torch.onnx.export(
        #         self,
        #         (
        #             self.dummy_nodes.to(self.device),
        #             self.dummy_loc.to(self.device),
        #             [edges.to(self.device) for edges in self.dummy_edges],
        #             self.dummy_vel.to(self.device),
        #             self.dummy_edge_attr.to(self.device),
        #         ),
        #         model_filename,
        #         opset_version=12,
        #     )
        #     wandb.save(model_filename)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]).to(self.device), torch.LongTensor(self.edges[1]).to(self.device)]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges




class VNDeepSets(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.prediction_mode = hyperparams.out_dim == 1
        self.model = "vndeepsets"
        self.hidden_dim = hyperparams.hidden_dim
        self.layer_pooling = hyperparams.layer_pooling
        self.final_pooling = hyperparams.final_pooling
        self.num_layers = hyperparams.num_layers
        self.nonlinearity = hyperparams.nonlinearity
        self.canon_feature = "p" # hyperparams.canon_feature
        self.canon_translation = hyperparams.canon_translation
        self.angular_feature = hyperparams.angular_feature
        self.dropout = hyperparams.dropout
        self.out_dim = hyperparams.out_dim
        self.in_dim = len(self.canon_feature)
        self.first_set_layer = VNDeepSetLayer(
            self.in_dim, self.hidden_dim, self.nonlinearity, self.layer_pooling, False, dropout=self.dropout
        )
        self.set_layers = SequentialMultiple(
            *[
                VNDeepSetLayer(
                    self.hidden_dim, self.hidden_dim, self.nonlinearity, self.layer_pooling, dropout=self.dropout
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.output_layer = (
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.batch_size = hyperparams.batch_size

        self.dummy_input = torch.zeros(1, device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros(1, device=self.device, dtype=torch.long)

    def forward(self, loc, edges):
        nb_particles = loc.shape[0]
        batch_indices = torch.arange(self.batch_size, device=self.device).reshape(-1, 1)
        batch_indices = batch_indices.repeat(1, nb_particles).reshape(-1)
        mean_loc = ts.scatter(loc, batch_indices, 0, reduce=self.layer_pooling)
        mean_loc = mean_loc.repeat(nb_particles, 1, 1).transpose(0, 1).reshape(-1, 3)
        canonical_loc = loc - mean_loc
        if self.canon_feature == "p":
            features = torch.stack([canonical_loc], dim=2) 
        # if self.canon_feature == "pv":
        #     features = torch.stack([canonical_loc, vel], dim=2)
        # elif self.canon_feature == "pva":
        #     angular = torch.linalg.cross(canonical_loc, vel, dim=1)
        #     features = torch.stack([canonical_loc, vel, angular], dim=2)
        # elif self.canon_feature == "pvc":
        #     features = torch.stack([canonical_loc, vel, canonical_loc * charges], dim=2)
        # elif self.canon_feature == "pvac":
        #     angular = torch.linalg.cross(canonical_loc, vel, dim=1)
        #     features = torch.stack([canonical_loc, vel, angular, canonical_loc * charges], dim=2)

        x, _ = self.first_set_layer(features, edges)
        x, _ = self.set_layers(x, edges)

        if self.prediction_mode:
            output = self.output_layer(x)
            output = output.squeeze()
            return output
        else:
            x = ts.scatter(x, batch_indices, 0, reduce=self.final_pooling)
        output = self.output_layer(x)

        output = output.repeat(nb_particles, 1, 1, 1).transpose(0, 1)
        output = output.reshape(-1, 3, 4)

        rotation_vectors = output[:, :, :3]
        translation_vectors = output[:, :, 3:] if self.canon_translation else 0.0
        translation_vectors = translation_vectors + mean_loc[:, :, None]

        return rotation_vectors, translation_vectors.squeeze()




