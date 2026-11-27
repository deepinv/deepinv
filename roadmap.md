<a id="roadmap"></a>

# Roadmap

Features that we would like to include in the near future (end 2026-mid 2027):

- Real data domain-specific I/O and examples: More real medical imaging, microscopy, remote sensing examples, domain-specific train/test datasets, raw data/geometry loaders and pretrained models. See [#870](https://github.com/deepinv/deepinv/pull/870), [#1062](https://github.com/deepinv/deepinv/pull/1062), [#1301](https://github.com/deepinv/deepinv/pull/1301), [#1361](https://github.com/deepinv/deepinv/pull/1361), [#1372](https://github.com/deepinv/deepinv/pull/1372), [#1467](https://github.com/deepinv/deepinv/pull/1467),
- Extend PhysicsGenerator to every physics: one of the key features of deepinv is the ability to generate physics on-the-fly, but this is currently limited to a few imaging modalities.
- Continue keeping up with state-of-the-art flow matching and diffusion models for inverse problems, with pretrained models, examples, and benchmarks. See [#1062](https://github.com/deepinv/deepinv/pull/1062), [#1470](https://github.com/deepinv/deepinv/pull/1470).
- Refactor optimization. *We heard the feedback*: incorporating new optimizers is not easy due to the currently unified design - we will refactor to expose the optimization iterates, simplifying the definition of new optimizers and allowing for more flexible optimization schemes. See [#1147](https://github.com/deepinv/deepinv/pull/1147), [#1255](https://github.com/deepinv/deepinv/pull/1255), [#1322](https://github.com/deepinv/deepinv/pull/1322), [#1375](https://github.com/deepinv/deepinv/pull/1375).
- Benchmarking and pretrained models: NTIRE-level image restoration models and benchmarks, and reference benchmarks in tomography, microscopy and medical imaging.
- Advanced microscopy operators: light-sheet PSFs, 4D STEM ptychography, electron tomography, SIM. See [#1418](https://github.com/deepinv/deepinv/pull/1418), [#1489](https://github.com/deepinv/deepinv/pull/1489).
- Advanced medical imaging operators: focus on ultrasound problems, motion simulation in MRI and CT. See [#1355](https://github.com/deepinv/deepinv/pull/1355), [#1465](https://github.com/deepinv/deepinv/pull/1465).
- Support out-of-the-box parameter efficient finetuning techniques: LoRA, gain-tuning, etc., with supervised or self-supervised losses.
- MultiGPU support: extending the deepinv.distributed module to support multiGPU evaluation of forward operators and reconstruction models. See [#1088](https://github.com/deepinv/deepinv/pull/1088).
- Agentic support: skills and tools for agents to set up data pipelines, and evaluate and finetune models for inverse problems. See [#1282](https://github.com/deepinv/deepinv/pull/1282), [#1362](https://github.com/deepinv/deepinv/pull/1362).
