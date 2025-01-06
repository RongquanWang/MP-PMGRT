
import torch
import torch.nn as nn


def bell_loss(p, y, *argv):
    y_p = torch.pow((y - p), 2)
    y_p_div = -1.0 * torch.div(y_p, 162.0)
    exp_y_p = torch.exp(y_p_div)
    loss = 300 * (1.0 - exp_y_p)
    loss = torch.mean(loss)
    return loss

def logcosh(p, y, *argv):
    loss = torch.log(torch.cosh(p - y))
    return torch.mean(loss)

def rmse(p, y, *argv):
    return torch.sqrt(nn.MSELoss()(p, y))

def GL(p, y, lam, eps=600, sigma=8):
    gl = eps / (lam ** 2) * (1 - torch.exp(-1 * ((y - p) ** 2) / (sigma ** 2)))
    gl = gl.mean()
    return gl

def rmse_bell(p, y, *argv):
    return rmse(p, y) + bell_loss(p, y)

def rmse_logcosh(p, y, *argv):
    return rmse(p, y) + logcosh(p, y)

def rmse_GL(p, y, *argv):
    return rmse(p, y) + GL(p, y, *argv)

def rmse_bell_logcosh(p, y, *argv):
    return rmse(p, y) + bell_loss(p, y) + logcosh(p, y)

def rmse_bell_GL(p, y, *argv):
    return rmse(p, y) + bell_loss(p, y) + GL(p, y, *argv)

def bell_logcosh(p, y, *argv):
    return bell_loss(p, y) + logcosh(p, y)

def bell_GL(p, y, *argv):
    return bell_loss(p, y) + GL(p, y, *argv)

def bell_logcosh_GL(p, y, *argv):
    return bell_loss(p, y) + logcosh(p, y) + GL(p, y, *argv)

def logcosh_GL(p, y, *argv):
    return logcosh(p, y) + GL(p, y, *argv)