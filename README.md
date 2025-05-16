# Visual Question Answering

This project involves creating a multiple-choice Visual Question Answering (VQA) dataset using the **Amazon Berkeley Objects (ABO)** dataset, evaluating baseline models, fine-tuning using **Low-Rank Adaptation (LoRA)**, and assessing performance using standard metrics.

## How to Run the Code

1. Create a new environment using Python 3.9.
2. Install the dependencies using the [`requirements.txt`](https://github.com/truptikhodwe/Visual-Question-Answering/blob/main/requirements.txt) file.
3. Run the [`inference.py`](https://github.com/truptikhodwe/Visual-Question-Answering/blob/main/inference.py) file.

```bash
conda create --name myEnv python=3.9 (if not already created)
conda activate myEnv
pip install -r requirements.txt
python inference.py --image_dir <PATH-TO-IMAGE-DIR> --csv_path <PATH-TO-IMAGE_METADATA-CSV>
```

Please refer to the [report.pdf](https://github.com/truptikhodwe/Visual-Question-Answering/blob/main/report.pdf) file for more details about:
- The models used
- Files and code structure
- Steps followed
- Final results of the project
