"""
reference: https://github.com/sanatonek/TNC_representation_learning/blob/master/tnc/tnc.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.types_ import *


class Discriminator(nn.Module):
    def __init__(self, input_size: int, w: float = 0.0):
        """
        :param input_size: size of the representation
        :param w:
        """
        super().__init__()
        self.input_size = input_size
        self.w = w

        self.model = nn.Sequential(nn.Linear(2*self.input_size, 4*self.input_size),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(4*self.input_size, 1))
        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor, x_tild: Tensor) -> Tensor:
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        out = self.model(x_all)
        return out

    def loss_function(self,
                      y: Tensor,
                      y_l: Tensor,
                      y_k: Tensor,
                      use_diag_loss: bool,
                      loss_hist: dict) -> Tensor:
        """
        :param y: reference repr
        :param y_l: neighboring repr
        :param y_k: non-neighboring repr
        :return:
        """
        d_p = self.forward(y, y_l)
        d_n = self.forward(y, y_k)

        neighbors = torch.ones(d_p.shape).to(y.device)
        non_neighbors = torch.zeros(d_n.shape).to(y.device)

        p_loss = self.criterion(d_p, neighbors)
        n_loss = self.criterion(d_n, non_neighbors)
        n_loss_u = self.criterion(d_n, neighbors)
        loss = p_loss + (1 - self.w) * n_loss + self.w * n_loss_u  # original TNC loss
        loss = loss / 2

        p_acc = torch.sum(nn.Sigmoid()(d_p) > 0.5).item() / d_p.shape[0]
        n_acc = torch.sum(nn.Sigmoid()(d_n) < 0.5).item() / d_n.shape[0]

        norm_y = F.normalize(y - y.mean(dim=1, keepdim=True), p=2, dim=1)  # (B, D)
        norm_y_l = F.normalize(y_l - y_l.mean(dim=1, keepdim=True), p=2, dim=1)  # (B, D)
        norm_y_k = F.normalize(y_k - y_k.mean(dim=1, keepdim=True), p=2, dim=1)  # (B, D)

        corr_pos = torch.mm(norm_y, norm_y_l.T)  # (B, B)
        corr_neg = torch.mm(norm_y, norm_y_k.T)  # (B, B)
        diag_pos = torch.diag(corr_pos, 0)  # (B,)
        diag_neg = torch.diag(corr_neg, 0)  # (B,)
        diag_pos_loss = torch.mean((1. - diag_pos)**2)
        diag_neg_loss = torch.mean((0. - diag_neg)**2)
        if use_diag_loss:
            loss += (diag_pos_loss + diag_neg_loss) / 2

        # log
        loss_hist['TNC/tnc_loss'] = loss
        loss_hist['TNC/p_acc'] = p_acc
        loss_hist['TNC/n_acc'] = n_acc
        loss_hist['TNC/acc'] = (p_acc + n_acc) / 2
        loss_hist['TNC/diag_pos_loss'] = diag_pos_loss
        loss_hist['TNC/diag_neg_loss'] = diag_neg_loss

        return loss


if __name__ == '__main__':
    # toy dataset
    B = 4
    x = torch.rand(B, 512)
    x_tild = torch.rand(B, 512)

    # forward
    D = Discriminator(512)
    out = D(x, x_tild)
    print('out.shape:', out.shape)
    print('out:', out)
