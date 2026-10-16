
# What Recurrence Can and Cannot Remember

AIGC5500 - Lab 04

## Task
Compare RNN, LSTM and GRU models on a delayed-recall task.

## Prediction (written before clipping experiment)

I predict that gradient clipping will not significantly improve
the plain RNN's accuracy at sequence length 30.

The gradient measurements in Part 2 show that gradients
become extremely small as they travel backward through time.

Gradient clipping limits large gradients, but it does not
restore gradients that have already vanished.

Therefore, I expect the RNN to remain near chance accuracy
(0.25) with or without gradient clipping.


## Setup

Create a Python virtual environment and install dependencies.

```bash
python -m venv .venv
```

Windows activation:

```powershell
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run

```bash
python benchmark.py
python gradients.py
python fixes.py
python timing.py
```

## Part 1: Benchmark Results

The experiment compared five recurrent configurations
using sequence lengths 10, 30 and 60.

Each configuration was tested with seeds 0, 1 and 2.

| Model | Length | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|---|
| RNN | 10 | 0.483 | 1.000 | 0.495 | 0.659 |
| RNN | 30 | 0.248 | 0.251 | 0.264 | 0.254 |
| RNN | 60 | 0.250 | 0.240 | 0.233 | 0.241 |
| LSTM Default | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM Default | 30 | 0.248 | 0.251 | 0.255 | 0.251 |
| LSTM Default | 60 | 0.249 | 0.251 | 0.254 | 0.251 |
| LSTM Bias 3 | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM Bias 3 | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM Bias 3 | 60 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU Default | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU Default | 30 | 0.241 | 0.257 | 0.254 | 0.251 |
| GRU Default | 60 | 0.248 | 0.249 | 0.240 | 0.246 |
| GRU Bias 3 | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU Bias 3 | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU Bias 3 | 60 | 0.998 | 1.000 | 1.000 | 0.999 |

### Observation

The default recurrent models struggled with longer
sequences and performed near chance accuracy (0.25).

LSTM and GRU with gate bias 3 maintained nearly perfect
accuracy even at sequence length 60.

The RNN at length 10 showed seed sensitivity.
Seed 1 achieved 1.000, while seeds 0 and 2 remained
near 0.5.

## Part 2: Gradient Through Time

Gradient norms were measured over a 60-step sequence,
using five random seeds and taking the median.

| Steps Back | RNN | LSTM Default | LSTM Bias 3 |
|---|---|---|---|
| 0 | 1.44866574e+00 | 3.84907901e-01 | 3.01311553e-01 |
| 9 | 2.54532602e-03 | 2.40298407e-03 | 1.16556972e-01 |
| 29 | 9.52534851e-09 | 1.78412364e-07 | 8.66561830e-02 |
| 59 | 8.15311299e-17 | 1.55543072e-13 | 1.81265950e-01 |

![Gradient Through Time](gradient_plot.png)

### Observation

The RNN and default LSTM experienced severe gradient
decay across long sequences.

The LSTM with forget-gate bias 3 preserved stronger
gradients, helping explain its benchmark performance.

## Part 3: Controlled Fixes

### Forget-Gate Bias Sweep

| Bias | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 0 | 0.247 | 0.251 | 0.258 | 0.252 |
| 1 | 0.495 | 0.496 | 1.000 | 0.664 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 |

Increasing the forget-gate bias improved delayed recall.

Bias 1 showed variability across seeds, while biases
2 and 3 achieved perfect accuracy.

### Gradient Clipping Comparison

| Clipping | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| On | 0.248 | 0.251 | 0.264 | 0.254 |
| Off | 0.248 | 0.251 | 0.264 | 0.254 |

Clipping produced no measurable difference in final
accuracy for this experiment.

This supports the earlier prediction because clipping
limits exploding gradients but does not directly solve
vanishing gradients.

### Gate-Bias Verification

The LSTM forget gate and GRU update gate occupy slice
[H:2H] in PyTorch's recurrent bias vectors.

Both bias_ih_l0 and bias_hh_l0 were set to 1.5 for
gate bias 3, producing an effective bias of 3.0.

The effective gate-bias values were printed and checked.

LSTM: [3.0, 3.0, 3.0, 3.0, 3.0]

GRU: [3.0, 3.0, 3.0, 3.0, 3.0]

## Part 4: LSTM Timing

The forward and backward passes were timed using
three repetitions per sequence length.

The best time was recorded.

| Sequence Length | Best Time (Seconds) |
|---|---|
| 50 | 0.002516 |
| 100 | 0.003588 |
| 200 | 0.017655 |
| 400 | 0.029679 |

Execution time generally increased with sequence length.

## Conclusion

The experiments demonstrate that recurrent networks
can struggle to preserve information over long sequences.

Gates provide a mechanism for retaining information,
but their initial values strongly influence learning.

The LSTM and GRU with gate bias 3 achieved nearly
perfect accuracy on the longest sequences.

Gradient clipping did not improve the plain RNN
because its main problem was vanishing gradients,
not exploding gradients.

Two important limitations of recurrence were observed:

1. Long-range dependencies can cause vanishing gradients.
   Direct connections, such as attention, provide shorter
   paths between distant sequence positions.

2. Sequential processing limits computational parallelism.
   Attention-based architectures can process sequence
   positions more concurrently.

## Known Limitations

The plain RNN starter test with seed 0 achieved only
0.483 accuracy, below the expected 0.9 threshold.

Changing to seed 1 achieved 1.000 accuracy.

The benchmark results were retained without modification.

Timing results were measured on one computer and may
vary depending on hardware and system load.
