# Adaptive Replay Buffer (ARB) for Offline-to-Online Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-AISTATS_2026-b31b1b.svg)](https://github.com/song970407/ARB)
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/song970407/ARB/blob/main/LICENSE)

Official implementation of **"Adaptive Replay Buffer for Offline-to-Online Reinforcement Learning"**, published at AISTATS 2026.

---

## Getting Started

This codebase is based on [CORL (Clean Offline Reinforcement Learning)](https://github.com/tinkoff-ai/CORL) by Tarasov et al. (2022). We adapted and extended the CORL framework to implement ARB on top of Cal-QL, PEX, and FamO2O.

### Installation

```bash
git clone https://github.com/song970407/ARB.git && cd ARB
pip install -r requirements/requirements_dev.txt
```

**Requirements**: Python 3.9, CUDA 11.3, PyTorch 1.11, MuJoCo

---

## Running Experiments

### Offline Pre-training
```bash
# Cal-QL + ARB on hopper-medium-v2
python algorithms/pretrain/cal_ql.py --env hopper-medium-v2
```
### Online Fine-tuning with ARB
```bash
# Cal-QL + ARB on hopper-medium-v2
python algorithms/finetune/cal_ql.py --env hopper-medium-v2 --replay_buffer arb
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{song2025adaptive,
  title={Adaptive Replay Buffer for Offline-to-Online Reinforcement Learning},
  author={Song, Chihyeon and Lee, Jaewoo and Park, Jinkyoo},
  journal={arXiv preprint arXiv:2512.10510},
  year={2025}
}
```

---

## Acknowledgements

This codebase is built upon [CORL (Clean Offline Reinforcement Learning)](https://github.com/tinkoff-ai/CORL):

```bibtex
@inproceedings{tarasov2022corl,
  title     = {{CORL}: Research-Oriented Deep Offline Reinforcement Learning Library},
  author    = {Tarasov, Denis and Nikulin, Alexander and Akimov, Dmitry and Kurenkov, Vladislav and Kolesnikov, Sergey},
  booktitle = {3rd Offline RL Workshop: Offline RL as a "Launchpad"},
  year      = {2022}
}
```

PEX and FamO2O implementations are adapted from their respective official codebases:
- [PEX](https://github.com/zhaoyi11/pex) (Zhang et al., 2023)
- [FamO2O](https://github.com/LeapLabTHU/FamO2O) (Wang et al., 2023)
