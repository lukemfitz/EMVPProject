# Benchmark Results — EMVP-Protected ResNet vs Plaintext

**Hardware:** Apple M1 Max (CPU, C extension)  
**Model:** ResNet (n_blocks=2, 16→32→64 channels)  
**Images:** 1000 test images per dataset  
**Weights:** Pretrained (`mnist_weights.npy`, `cifar10_weights.npy`)

---

## Accuracy

| Metric | MNIST | CIFAR-10 |
|---|---|---|
| Plaintext accuracy | 99.7% (997/1000) | 90.2% (902/1000) |
| EMVP accuracy | 99.7% (997/1000) | 90.1% (901/1000) |
| Accuracy loss from encryption | 0.0% | 0.1% |
| Prediction agreement | 100.0% (1000/1000) | 99.9% (999/1000) |

## Logit Error (quantisation noise)

| Metric | MNIST | CIFAR-10 |
|---|---|---|
| Mean L2 logit error | 0.0208 | 0.1141 |
| Max L2 logit error | 0.0648 | 0.2542 |

## Speed

| Metric | MNIST | CIFAR-10 |
|---|---|---|
| Plaintext | 8.86 ms/image | 14.27 ms/image |
| EMVP (encrypted) | 55 ms/image | 84 ms/image |
| Slowdown (EMVP / plaintext) | 6× | 6× |
