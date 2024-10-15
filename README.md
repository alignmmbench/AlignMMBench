# AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models


<p align="center">
    <img src="./assets/index.png" width="96%" height="50%">
</p>

---

## 🔥 News

* **`2024.09.23`** 🌟 We provide the code, model, and data for evaluation!
* **`2024.06.14`** 🌟 We released AlignMMBench, a comprehensive alignment benchmark for vision language models!


## 👀 Introduce to AlignMMBench

AlignMMBench a multimodal alignment benchmark that encompasses both single-turn and multi-turn dialogue scenarios. It includes three categories and thirteen capability tasks, with a total of 4,978 question-answer pairs.

### Features

1. **High-Quality Annotations**: Reliable benchmark with meticulous human annotation and multi-stage quality control processes.

2. **Self Critic**: To improve the controllability of alignment evaluation, we introduce the CritiqueVLM, a ChatGLM3-6B based evaluator that has been rule-calibrated and carefully finetuned. With human judgements, its evaluation consistency surpasses that of GPT-4.
   
3. **Diverse Data**: Three categories and thirteen capability tasks, including both single-turn and multi-turn dialogue scenarios.

<img src="./assets/image_examples.png" width="100%" height="50%">

## 💻 Evaluate your model

**Step 0** 
Download AlignMMBench data from [hidden], and CritiqueVLM model file from [hidden].

> Due to the requirements of double-blind review, we have hidden the download links for the data and models. They will be made public after the review process is completed.

**Step 1** 
Infer your model on AlignMMBench and get your model responses in `.jsonl` format like this:
```json
{"question_id": "00000000-0", "predict": "..."}
{"question_id": "00000000-1", "predict": "..."}
{"question_id": "00000000-2", "predict": "..."}
```

**Step 2** Clone this repository and install requirements.
```bash
git clone https://github.com/alignmmbench/AlignMMBench.git && cd AlignMMBench
pip install -r requirements.txt
```

**Step 3** Run CritiqueVLM evaluator in `evaluate.py`:
```bash
python evaluate.py --critic_model_path <critiqueVLM_path> --response_file <your_model_responses_path> --metadata_file <metadata_path> --save_path <path_to_save_detailed_evaluation_results>
```


## 📈 Results

<p align="center">
    <img src="./assets/leaderboard.png" width="96%" height="50%">
</p>

## License

The use of the dataset and the original videos is governed by the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0) license, as detailed in the  [LICENSE](./LICENSE).

