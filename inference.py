import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Define Huggingface LoRA model path
    lora_model_path = "RishabhD04/VQA_Fine-Tuned"
    base_model = "Salesforce/blip-vqa-capfilt-large"

    # Load processor and base model
    processor = BlipProcessor.from_pretrained(base_model)
    config = PeftConfig.from_pretrained(lora_model_path)
    model = BlipForQuestionAnswering.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, lora_model_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load metadata
    df = pd.read_csv(args.csv_path)

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs)
            answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            answer = "error"
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
