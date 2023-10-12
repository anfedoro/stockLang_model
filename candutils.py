# Common functions to use for candles profiling/classification (no DL, simple analitic approach)

import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def CandlesClassification(dataset: torch.Tensor):
    r"""CandleClassification function, analytically classifies candles on 24 variuus class.

    Input is torch tensor with 4 columns, respectively 'Open', 'High', 'Low', 'Close'
    Returns  new tensor with candle class set
    """
    # canldes parameters
    o = dataset[:, 0].view(-1, 1)
    h = dataset[:, 1].view(-1, 1)
    l = dataset[:, 2].view(-1, 1)
    c = dataset[:, 3].view(-1, 1)

    # empty classification tensor
    cond = torch.zeros(len(o), 1)

    # Three types of candles
    doji = o == c
    green = o < c
    red = o > c

    # Four types of candles bodies
    body25 = (abs(o - c) / (h - l) > 0) & (abs(o - c) / (h - l) <= 0.25)
    body75 = (abs(o - c) / (h - l) > 0.25) & (abs(o - c) / (h - l) <= 0.75)
    body00 = (abs(o - c) / (h - l) > 0.75) & (abs(o - c) / (h - l) <= 1)

    # Three subsection of candle height
    p0 = l.view(-1, 1)
    p1 = (h + l) * 0.5
    p2 = h.view(-1, 1)

    # mid point of candle body
    mid = (o + c) / 2

    # mid point match to one of three subsection
    mid1 = (mid >= p0) & (mid <= p1)
    mid2 = (mid > p1) & (mid <= p2)

    # set of conditions to define a canlde class/type
    cond[:, [0]] = torch.where((doji & mid1), 0.0, cond[:, [0]])
    cond[:, [0]] = torch.where((doji & mid2), 1.0, cond[:, [0]])

    cond[:, [0]] = torch.where((green & body25 & mid1), 2.0, cond[:, [0]])
    cond[:, [0]] = torch.where((green & body25 & mid2), 3.0, cond[:, [0]])

    cond[:, [0]] = torch.where((green & body75 & mid1), 4.0, cond[:, [0]])
    cond[:, [0]] = torch.where((green & body75 & mid2), 5.0, cond[:, [0]])

    cond[:, [0]] = torch.where((green & body00), 6.0, cond[:, [0]])

    cond[:, [0]] = torch.where((red & body25 & mid1), 7.0, cond[:, [0]])
    cond[:, [0]] = torch.where((red & body25 & mid2), 8.0, cond[:, [0]])

    cond[:, [0]] = torch.where((red & body75 & mid1), 9.0, cond[:, [0]])
    cond[:, [0]] = torch.where((red & body75 & mid2), 10.0, cond[:, [0]])

    cond[:, [0]] = torch.where((red & body00), 11.0, cond[:, [0]])

    return torch.concat([dataset, cond], dim=1)


def CandlesClass(dataset: torch.Tensor):
    # canldes parameters
    o = dataset[:, 0].view(-1, 1)
    h = dataset[:, 1].view(-1, 1)
    l = dataset[:, 2].view(-1, 1)
    c = dataset[:, 3].view(-1, 1)

    # empty classification tensor
    cond = torch.zeros(len(o), 1)

    # Three types of candles
    doji = abs(o - c) / (h - l) < 0.25
    green = o < c
    red = o > c

    # Four types of candles bodies

    body50 = (abs(o - c) / (h - l) >= 0.25) & (abs(o - c) / (h - l) < 0.50)
    body00 = (abs(o - c) / (h - l) >= 0.50) & (abs(o - c) / (h - l) <= 1)

    # set of conditions to define a canlde class/type
    cond[:, [0]] = torch.where((doji), 0.0, cond[:, [0]])

    cond[:, [0]] = torch.where((green & body50), 1.0, cond[:, [0]])
    cond[:, [0]] = torch.where((green & body00), 2.0, cond[:, [0]])

    cond[:, [0]] = torch.where((red & body50), 3.0, cond[:, [0]])
    cond[:, [0]] = torch.where((red & body00), 4.0, cond[:, [0]])

    return torch.concat([dataset, cond], dim=1)


# My own Normalisation function. the only reason I use them is I can easie reproduce them in TradingView Pinescript,
# while e.g Pytorch nn.instanceNorm1d, gives sllughtly different result. But technically there is no differece which one to use here.
def MinMaxNorm(x: torch.Tensor, dim=None, shift=False):
    r"""MinMax Normalization function, normalize values in (0,1) range.

    dim - dimentions setting,
    where None - for global normalization, 1 - for per instance normalizaton and 0 - per feature normalization
    shif = True shifts all values agains candle Open"""

    if dim == None:
        return (x - x.min()) / (x.max() - x.min())
    elif dim == 1:
        if shift:
            norm_x = torch.nan_to_num(
                (x - x.min(dim).values.view(-1, 1))
                / (x.max(dim).values - x.min(dim).values).view(-1, 1),
                0,
            )
            return norm_x - norm_x[:, 0].view(-1, 1)
        else:
            return torch.nan_to_num(
                (x - x.min(dim).values.view(-1, 1))
                / (x.max(dim).values - x.min(dim).values).view(-1, 1),
                0,
            )

    elif dim == 0:
        return (x - x.min(dim).values) / (x.max(dim).values - x.min(dim).values)


def StandardNorm(x: torch.Tensor, dim=None):
    r"""Standard Normalization function, normalize values in (0,1) range.

    dim - dimentions setting,
    where None - for global normalization, 1 - for per instance normalizaton and 0 - per feature normalization
    """

    if dim == None:
        return (x - x.mean()) / (x.std())
    elif dim == 1:
        return (x - x.mean(dim).reshape(-1, 1)) / (x.std(dim).reshape(-1, 1))
    else:
        return (x - x.mean(dim)) / (x.std(dim))


##Function to plot candles from the dataset generated based on prediction
def plot_sample(sample, plot_class=False):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=torch.arange(len(sample)).numpy(),
                open=sample[:, 0].numpy(),
                high=sample[:, 1].numpy(),
                low=sample[:, 2].numpy(),
                close=sample[:, 3].numpy(),
            )
        ]
    )

    if plot_class:
        fig.add_trace(
            go.Scatter(
                x=torch.arange(len(sample)).numpy(),
                y=sample[:, 1].numpy(),
                mode="text",
                text=sample[:, 4].numpy(),
                textposition="top center",
                textfont={"size": 9},
            )
        )

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()
