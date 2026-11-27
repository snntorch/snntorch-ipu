"""Run the README example and check that one training step updates the weights."""

import json
import torch
from torch import nn
import snntorch as snn
from snntorch import spikegen, surrogate, utils


def main():
    torch.manual_seed(101)
    torch.set_num_threads(1)
    net = nn.Sequential(
        nn.Conv2d(1, 8, 5),
        nn.MaxPool2d(2),
        snn.Leaky(0.5, init_hidden=True, spike_grad=surrogate.fast_sigmoid()),
        nn.Conv2d(8, 16, 5),
        nn.MaxPool2d(2),
        snn.Leaky(0.5, init_hidden=True, spike_grad=surrogate.fast_sigmoid()),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
        snn.Leaky(
            0.5, init_hidden=True, spike_grad=surrogate.fast_sigmoid(), output=True
        ),
    )
    data = torch.rand(25, 1, 1, 28, 28)
    spikes = torch.stack([net(x)[0] for x in data])
    assert spikes.shape == (25, 1, 10)
    utils.reset(net)
    encoded = spikegen.rate(torch.tensor([[0.25, 0.75]]), num_steps=8)
    linear = nn.Linear(2, 2)
    lif = snn.Leaky(0.5, spike_grad=surrogate.fast_sigmoid())
    optimiser = torch.optim.SGD(linear.parameters(), lr=0.01)
    before = linear.weight.detach().clone()
    mem = torch.zeros(1, 2)
    outputs = []
    for x in encoded:
        spk, mem = lif(linear(x), mem)
        outputs.append(spk)
    loss = (torch.stack(outputs).mean() - 0.5).square()
    optimiser.zero_grad()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in linear.parameters()
    )
    optimiser.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, linear.weight)
    print(
        json.dumps(
            dict(
                seed=101,
                readme_shape=list(spikes.shape),
                readme_spikes=int(spikes.sum()),
                loss=float(loss.detach()),
                gradient_norm=float(linear.weight.grad.norm()),
                weights_changed=True,
                package_file=snn.__file__,
                torch=torch.__version__,
            )
        )
    )


if __name__ == "__main__":
    main()
