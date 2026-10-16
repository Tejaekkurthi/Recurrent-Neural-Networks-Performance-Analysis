
import time
import torch
import torch.nn as nn

# Part 4: LSTM Timing Experiment

torch.manual_seed(0)

sequence_lengths = [50, 100, 200, 400]

print("LSTM Forward and Backward Timing")
print("-" * 40)
print(f"{'Length':<12} {'Best Time (seconds)':<20}")

for seq_len in sequence_lengths:
    times = []

    model = nn.LSTM(
        input_size=8,
        hidden_size=32,
        batch_first=True
    )

    x = torch.randn(16, seq_len, 8)

    # Warm-up
    output, _ = model(x)
    output[:, -1, :].sum().backward()

    for repeat in range(3):
        model.zero_grad(set_to_none=True)

        start = time.perf_counter()

        output, _ = model(x)
        loss = output[:, -1, :].sum()
        loss.backward()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    best_time = min(times)

    print(f"{seq_len:<12} {best_time:<20.6f}", flush=True)
