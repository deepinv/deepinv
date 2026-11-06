.. _roadmap:

Roadmap
=========

Features that we would like to include in the near future (end 2026-mid 2027):

- Real data domain-specific I/O and examples: More real medical imaging, microscopy, remote sensing examples, domain-specific train/test datasets, raw data/geometry loaders and pretrained models. See :gh:`870`, :gh:`1062`, :gh:`1301`, :gh:`1361`, :gh:`1372`, :gh:`1467`,
- Extend PhysicsGenerator to every physics: one of the key features of deepinv is the ability to generate physics on-the-fly, but this is currently limited to a few imaging modalities.
- Continue keeping up with state-of-the-art flow matching and diffusion models for inverse problems, with pretrained models, examples, and benchmarks. See :gh:`1062`, :gh:`1470`.
- Refactor optimization. *We heard the feedback*: incorporating new optimizers is not easy due to the currently unified design - we will refactor to expose the optimization iterates, simplifying the definition of new optimizers and allowing for more flexible optimization schemes. See :gh:`1147`, :gh:`1255`, :gh:`1322`, :gh:`1375`.
- Benchmarking and pretrained models: NTIRE-level image restoration models and benchmarks, and reference benchmarks in tomography, microscopy and medical imaging.
- Advanced microscopy operators: light-sheet PSFs, 4D STEM ptychography, electron tomography, SIM. See :gh:`1418`, :gh:`1489`.
- Advanced medical imaging operators: focus on ultrasound problems, motion simulation in MRI and CT. See :gh:`1355`, :gh:`1465`.
- Support out-of-the-box parameter efficient finetuning techniques: LoRA, gain-tuning, etc., with supervised or self-supervised losses.
- MultiGPU support: extending the deepinv.distributed module to support multiGPU evaluation of forward operators and reconstruction models. See :gh:`1088`.
- Agentic support: skills and tools for agents to set up data pipelines, and evaluate and finetune models for inverse problems. See :gh:`1282`, :gh:`1362`.