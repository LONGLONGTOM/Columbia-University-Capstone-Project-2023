# JP Morgan: Measuring Behavior of Language Model and Effect of Teacher model enabled CoT Reasoning Fine-tuning

### Group members Name UNI 
- Haolong Liu (hl3614) (Team captain)
- Yudu Chen (yc4142)
- Jincheng Liu (jl6298)
- William Gu(wg2400)
- Yuanyi Hu (yh3506)


Emails  &lt;UNI&gt; @ columbia.edu

**Accenture mentor & co-mentors:** Akshat Gupta, Simerjot Kaur

**Instructor/CA:** Sining Chen, Vivian Zhang, Adam Kelleher, Savannah Thais

At the begining of the project, we actually changed our research topic. Originally, the topic was "Do Large Language Model Has a Personality?", but the mentor instructed us to study something else. 
Our study focuses on a large language model with a relatively small size to explore these aspects. The primary challenge lies in developing an efficient training methodology that enhances the model's comprehension abilities and ensures proficiency in demonstrating these complex behaviors. The importance of this research stems from the need to understand and responsibly harness the capabilities of LLMs. By investigating a smaller model, we aim to contribute valuable insights into these models' ethical implications and practical capabilities. This understanding is crucial for the responsible development and deployment of AI technologies in societal contexts, where moral considerations are as important as technical advancements. Our study extends the investigation into the impact of a large language model's reasoning teaching, particularly employing the Chain of Thought (CoT) approach, on the model's bias, ethics, reasoning, and creativity. This exploration is necessary and meaningful, considering the increasing integration of LLMs in diverse fields. The subsequent sections of this paper detail our comprehensive approach, encompassing the generation and utilization of data with the CoT approach, the fine-tuning methodologies employed, and the evaluation metrics of the model across various dimensions. This study aims to contribute to the understanding of AI ethics and capabilities.

The problem of understanding the behaviors, including bias, ethics, reasoning, and creativity, exhibited by large language models (LLMs) is crucial given their widespread applications in contemporary society. This study aims to investigate the behavior of a relatively small language model with approximately 3 billion parameters. The challenge at hand is to find an efficient training methodology for the small language model, enabling it to comprehend and proficiently demonstrate the identified behaviors. Addressing this problem is essential for advancing our understanding of the ethical implications and capabilities of LLMs, contributing to the responsible development and deployment of artificial intelligence in various societal contexts.

**Directory tree**
```
│   .gitignore
│   CONTRIBUTING.md
│   README.md
│   requirements.txt
│
├───.ipynb_checkpoints
├───confidence_scores
│       analyze.ipynb
│       generate.ipynb
│       utils.py
│
├───data preprocessing
│   └───.ipynb_checkpoints
├───documentation
│       GCP.txt
│       README.md
│
├───EDA
│       audio_EDA.ipynb
        phenome_EDA.ipynb
│
├───preprocessing
│       audio_preprocess.py
│       preprocess examples.ipynb
│       README.md
│
├───wav2vec
│   │   add_noise.ipynb
│   │   downsample.ipynb
│   │   KR_wav2vec2_XLS_R_finetune.ipynb
│   │   plots.ipynb
│   │   template.ipynb
│   │   utils.py
│   │   wav2vec4g_accent.ipynb
│   │   wav2vec4g_experiments.ipynb
│   │   wav2vec4g_noisy.ipynb
│   │   wav2vec_accent.ipynb
│   │   wav2vec_languages.ipynb
│   │   wav2vec_noisy.ipynb
│   │
│   └───.ipynb_checkpoints
│           wav2vec4g_experiments-checkpoint.ipynb
│           wav2vec_confidence_score-checkpoint.ipynb
│
├───wav2vec_finetune
│       wav2vec_finetune_hebrew.ipynb
│       wav2vec_finetune_hebrew_eval.ipynb
│       wav2vec_hebrew_eval.ipynb
│
├───whisper
│   │   audio.mp3
│   │   requirements.txt
│   │   robust_comp_plt.ipynb
│   │   robust_downsample.ipynb
│   │   robust_noise.ipynb
│   │   utils.py
│   │   wer_df.csv
│   │   whisper_intro.ipynb
│   │   whisper_intro_noise.ipynb
│   │
│   └───.ipynb_checkpoints
│           robust_noise-checkpoint.ipynb
│           whisper_intro-checkpoint.ipynb
│
└───whisper_finetune
        README.md
        requirements.txt
        utils.py
        whisper_finetune_demo.ipynb
        whisper_finetune_eval.ipynb
        whisper_finetune_multilingual.ipynb
```

Linked to Lierature collection: https://docs.google.com/spreadsheets/d/1b1Y5JY26M0FAwn3q2rhoSoDdpPyCs_wcNxKMjWfgoH4/edit#gid=0

Linked to Google Drive: https://drive.google.com/drive/u/0/folders/1Q4stOpBOGKcRoVKYaS2bRvYxsDRXIUdG

