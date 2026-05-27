# Anchor-based Maximum Discrepancy

Reproducibility code for **Anchor-based Maximum Discrepancy for Relative
Similarity Testing**.

This work is done by
- Zhijian Zhou (unimelb) zhijianzhou.ml@gmail.com
- Liuhua Peng (unimelb) liuhua.peng@unimelb.edu.au
- Xunye Tian (unimelb) xunyetian.ml@gmail.com
- Feng Liu (unimelb) fengliu.ml@gmail.com

## Setup

The original experiments were run with Python 3.9 and the packages listed in
`environment.yml`.

```bash
conda env create -f environment.yml
conda activate AMD
mkdir -p Results/demos Results/DA Results/ADV Results/Dire Results/Reg
```

Most paper experiments are GPU-oriented and use CUDA by default. The scripts in
`Exp_code` also use output paths relative to their own folders, so run them from
the corresponding experiment directory, for example `cd Exp_code/DA` before
running a DA script.

## Quick Demo

Run a small CPU-only Gaussian relative-similarity demo:

```bash
python demos/gaussian_amd_rst.py
```

The demo uses three Gaussian distributions: anchor `U`, candidate `P`, and
candidate `Q`. AMD selects a bandwidth and a direction from pilot samples, then
uses an independent test sample with a wild-bootstrap threshold. It reports AMD
against fixed-P, fixed-Q, random-direction, and two-sided variants, which makes
the direction-learning effect visible without relying on a single unfavorable
baseline direction. All variants use the same selected bandwidth in each trial,
so the comparison isolates the testing direction. The CSV summary is written to
`Results/demos/`.

## Main Paper Experiments

### Figure 1: Benchmark Relative Similarity Testing

The baseline methods used for the MNIST/CIFAR10 benchmark are implemented in
`baseline_tests/`:

- `MMD_asym.py`: MMD-D and median-heuristic MMD variants.
- `KLFI.py`: kernelized LFI-style relative similarity test.
- `UME_asym.py` and `UME_wild.py`: UME baselines.
- `SCHE_LBI.py`: classifier-style baselines.
- `AutoTST.py`: AutoML classifier baseline.

The AMD implementation used by these comparisons is in `Exp_code/Reg/utils.py`
and `Exp_code/Dire/utils.py`. A single top-level Figure 1 sweep script is not
included in this repository; the provided files are the reusable method and
baseline components.

### Figure 2: Direction Inference Probability

These scripts evaluate whether Phase I infers the correct relative-similarity
direction `F`:

```bash
cd Exp_code/Dire
python MNIST.py
python CIFAR10_ddpm.py
```

Set `per=0.5` for the no-relative-difference/type-I setting; values below or
above `0.5` make `Q` or `P` closer to the anchor respectively.

For a quick synthetic check of the Phase I direction rule:

```bash
cd Exp_code/Dire
python phase1_direction_benchmark.py --datasets BLOB HDGM --sample_sizes 60 120 200 --reps 20
```

### Figure 3 and Regularization Studies

These scripts sweep the regularization coefficient used in the AMD optimization
and compare the original learned-direction test with the two-sided variant:

```bash
cd Exp_code/Reg
python MNIST.py
python CIFAR10_ddpm.py
```

### Figure 4: ImageNet Variant Relative Model Performance

The ImageNet feature experiments compare which ImageNet variant is closer to
the original ImageNet feature distribution:

```bash
cd Exp_code/DA
python v2_a.py
python a_sk.py
python r_v2.py
python sk_r.py
```

Required feature files in `Exp_code/DA/`:

- `imagenet_Fea.pt`
- `imagenetv2_Fea.pt`
- `imageneta_Fea.pt`
- `imagenetr_Fea.pt`
- `imagenetsk_Fea.pt`

### Figure 5: Adversarial Perturbation Detection

The adversarial relative-similarity experiment compares CIFAR10 perturbation
levels against a `4/255` reference perturbation:

```bash
cd Exp_code/ADV
python ADV.py --net resnet18 --model_path ./Res18_model/net_150.pth
```

If the checkpoint is unavailable, train the CIFAR10 model first:

```bash
cd Exp_code/ADV
python train_model.py --outf ./Res18_model
```

`ADV.py` now takes `--net`, `--dataset`, and `--model_path` directly and passes
them to the adversarial generator.

## Data Files

The synthetic Gaussian demo has no external dependencies. The main experiments
require dataset assets generated or downloaded separately:

- MNIST/CIFAR10 experiments use torchvision datasets plus generated samples such
  as `Fake_MNIST_data_EP100_N10000.pckl` and `ddpm_generated_images.npy`.
- Some baseline loaders also reference `cifar10_X_adversarial.npy` and
  `HIGGS_TST.pckl`.
- ImageNet experiments require precomputed ResNet feature tensors listed above.

## Implementation Notes

AMD has two phases in this codebase:

- Phase I learns the direction `F` with a practical studentized RST criterion
  implemented in `Exp_code/amd_core.py`. It searches a median-scaled
  Gaussian/Laplace bandwidth grid, includes a scale-free energy discrepancy,
  and uses a small bagged consensus to stabilize the sign. The experiment
  scripts use an independent Phase I pilot of at least 2048 samples when those
  samples are available or can be generated, while the downstream test still
  uses its requested sample size. The legacy optimization arguments are kept in
  the experiment scripts for compatibility, but direction selection no longer
  depends on running two fragile direction-specific optimizations.
- Phase II fixes the learned direction/kernel and tests an independent sample
  using a wild-bootstrap threshold. See `Ours_method` and `wildbootstrapWD2`.
  The Phase I direction is kept fixed during Phase II rather than being flipped
  by a failed downstream kernel optimization.

The adversarial utilities were cleaned so importing `train_model.py` or
`adv_generator.py` no longer parses command-line arguments or downloads data.
