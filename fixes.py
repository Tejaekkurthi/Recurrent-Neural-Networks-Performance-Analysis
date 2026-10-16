
from lab04_starter import train_recall

# Part 3: Controlled Fixes

seeds = [0, 1, 2]
seq_len = 30


# Experiment 1: LSTM Forget-Gate Bias Sweep

print("LSTM Forget-Gate Bias Sweep")
print("-" * 65)
print(f"{'Bias':<10} {'Seed 0':<10} {'Seed 1':<10} {'Seed 2':<10} {'Mean':<10}")

for bias in [0, 1, 2, 3]:
    accuracies = []

    for seed in seeds:
        accuracy, history = train_recall(
            "lstm",
            seq_len=seq_len,
            seed=seed,
            gate_bias=float(bias)
        )
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)

    print(
        f"{bias:<10} "
        f"{accuracies[0]:<10.3f} "
        f"{accuracies[1]:<10.3f} "
        f"{accuracies[2]:<10.3f} "
        f"{mean_accuracy:<10.3f}",
        flush=True
    )


# Experiment 2: Gradient Clipping Comparison

print("\nRNN Gradient Clipping Comparison")
print("-" * 65)
print(f"{'Clipping':<10} {'Seed 0':<10} {'Seed 1':<10} {'Seed 2':<10} {'Mean':<10}")

for name, clip_value in [("On", 1.0), ("Off", None)]:
    accuracies = []

    for seed in seeds:
        accuracy, history = train_recall(
            "rnn",
            seq_len=seq_len,
            seed=seed,
            clip=clip_value
        )
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)

    print(
        f"{name:<10} "
        f"{accuracies[0]:<10.3f} "
        f"{accuracies[1]:<10.3f} "
        f"{accuracies[2]:<10.3f} "
        f"{mean_accuracy:<10.3f}",
        flush=True
    )
