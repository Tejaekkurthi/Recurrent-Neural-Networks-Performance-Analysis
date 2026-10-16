
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Part 2: Gradient Through Time

def measure_gradients(kind, gate_bias=None):
    gradients = []

    for seed in range(5):
        torch.manual_seed(seed)

        hidden = 32

        if kind == "rnn":
            model = nn.RNN(8, hidden, batch_first=True)
        else:
            model = nn.LSTM(8, hidden, batch_first=True)

        # Set LSTM forget-gate bias
        if gate_bias is not None:
            with torch.no_grad():
                model.bias_ih_l0[hidden:2 * hidden].fill_(gate_bias / 2)
                model.bias_hh_l0[hidden:2 * hidden].fill_(gate_bias / 2)

        # Batch 16, sequence length 60, input features 8
        x = torch.randn(16, 60, 8, requires_grad=True)

        output, _ = model(x)

        # Loss reads only the last step
        loss = output[:, -1, :].sum()

        loss.backward()

        # Gradient norm at each input step
        norms = x.grad.norm(dim=2).mean(dim=0)

        gradients.append(norms.detach())

    # Median across five seeds
    return torch.stack(gradients).median(dim=0).values


rnn_grad = measure_gradients("rnn")
lstm_grad = measure_gradients("lstm")
lstm_bias_grad = measure_gradients("lstm", gate_bias=3.0)

# Plot gradients
steps_back = list(range(60))

plt.figure(figsize=(9, 5))

plt.semilogy(steps_back, rnn_grad.flip(0).numpy(), label="RNN")
plt.semilogy(steps_back, lstm_grad.flip(0).numpy(), label="LSTM Default")
plt.semilogy(steps_back, lstm_bias_grad.flip(0).numpy(), label="LSTM Bias 3")

plt.xlabel("Steps Back from Loss")
plt.ylabel("Gradient Norm (Log Scale)")
plt.title("Gradient Through Time")
plt.legend()
plt.grid(True)

plt.savefig("gradient_plot.png")
plt.show()

# Print required gradient values
print("Steps Back | RNN | LSTM Default | LSTM Bias 3")

for step in [0, 9, 29, 59]:
    print(
        step,
        f"{rnn_grad.flip(0)[step].item():.8e}",
        f"{lstm_grad.flip(0)[step].item():.8e}",
        f"{lstm_bias_grad.flip(0)[step].item():.8e}"
    )
