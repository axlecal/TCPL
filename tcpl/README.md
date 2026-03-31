# TCPL: Task-Conditioned Prompt Learning for Few-Shot Cross-Subject MI-EEG

This project provides a full PyTorch implementation of:

**TCPL: Task-Conditioned Prompt Learning for Few-Shot Cross-Subject Motor Imagery EEG Decoding**

It includes:
- TCPL model 
- Episode-based meta-training loop
- Subject-wise dataset and episode sampler
- Validation/test scripts
- Few-shot inference for unseen subjects
- Dummy dataset and end-to-end sanity check

## 1. Project Structure

```text
tcpl/
├── configs/
│   └── default.yaml
├── datasets/
│   ├── eeg_dataset.py
│   ├── episode_sampler.py
│   └── dummy_dataset.py
├── models/
│   ├── support_encoder.py
│   ├── prompt_generator.py
│   ├── tcn.py
│   ├── transformer.py
│   ├── classifier.py
│   └── tcpl_model.py
├── trainers/
│   └── meta_trainer.py
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── init.py
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── infer_subject.py
│   └── sanity_check.py
├── README.md
└── requirements.txt
```
