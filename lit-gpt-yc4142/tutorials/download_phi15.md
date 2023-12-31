## Download [phi-1.5](https://arxiv.org/abs/2309.05463) weights

A team at Microsoft Research has made available Phi 1.5, which is a 1.3 billion parameter model optimized for common sense reasoning in natural language, showing performance on par with models 5x its size, especially in grade-school mathematics and basic coding. This model retains characteristics of larger LLMs, and significant improvement was noted in reducing toxic and biased generations by avoiding web data. It's also worth highlighting that while this model performs well on language understanding and common sense reasoning tasks, it is a base model that has not undergone any supervised instruction finetuning or finetuning with RLHF.

The model was trained the same data sources (7B tokens) as its [phi-1](https://arxiv.org/abs/2306.11644) predecessor, which includes

- a Python code subset from [The Stack](https://arxiv.org/abs/2211.15533) v1.2
- Q&A texts from [StackOverflow](https://archive.org/download/stackexchange)
- code from DeepMind [code_contests](https://github.com/deepmind/code_contests)
- synthetic Python textbooks and exercises generated by [gpt-3.5-turbo-0301](https://platform.openai.com/docs/models/gpt-3-5)

In addition, to create phi-1.5, the authors included additional textbook-quality synthetic text (roughly 20B tokens) in natural language, which was created using the [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) approach.

The model weights are released under a [*Microsoft Research license*](https://huggingface.co/microsoft/phi-1_5/blob/main/README.md#license).

In order to use the phi-1.5 model checkpoint, which requires about 3 Gb of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id microsoft/phi-1_5

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/microsoft/phi-1_5
```

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/microsoft/phi-1_5
```
