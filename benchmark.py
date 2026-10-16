
from lab04_starter import train_recall

# Part 1: Benchmark recurrent networks

configurations = [
    ("RNN", "rnn", None),
    ("LSTM Default", "lstm", None),
    ("LSTM Bias 3", "lstm", 3.0),
    ("GRU Default", "gru", None),
    ("GRU Bias 3", "gru", 3.0)
]

sequence_lengths = [10, 30, 60]
seeds = [0, 1, 2]

print("Lab 04 - Recurrent Network Benchmark")
print("-" * 75)
print(f"{'Model':<18} {'Length':<10} {'Seed 0':<10} {'Seed 1':<10} {'Seed 2':<10} {'Mean':<10}")

for name, kind, gate_bias in configurations:
    for seq_len in sequence_lengths:
        accuracies = []

        for seed in seeds:
            accuracy, history = train_recall(
                kind,
                seq_len=seq_len,
                seed=seed,
                gate_bias=gate_bias
            )
            accuracies.append(accuracy)

        mean_accuracy = sum(accuracies) / len(accuracies)

        print(
            f"{name:<18} {seq_len:<10} "
            f"{accuracies[0]:<10.3f} "
            f"{accuracies[1]:<10.3f} "
            f"{accuracies[2]:<10.3f} "
            f"{mean_accuracy:<10.3f}",
            flush=True
        )
