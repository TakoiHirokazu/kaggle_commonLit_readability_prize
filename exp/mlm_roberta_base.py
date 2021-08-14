# ========================================
# library
# ========================================
import pandas as pd
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
import os


# ==================
# Constant
# ==================
ex = "_mlm_roberta_base"
TRAIN_PATH = "../data/train.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/mlm_roberta_base")
    os.makedirs(f"../output/ex/ex{ex}/chk")

TEXT_SAVE_PATH = f"../output/ex/ex{ex}/text.txt"
SAVE_PATH = f"../output/ex/ex{ex}/mlm_roberta_base"
CHK_PATH = f"../output/ex/ex{ex}/chk"


# ================================
# Main
# ================================
data = pd.read_csv(TRAIN_PATH)
data['excerpt'] = data['excerpt'].apply(lambda x: x.replace('\n', ''))
text = '\n'.join(data.excerpt.tolist())
with open(TEXT_SAVE_PATH, 'w') as f:
    f.write(text)

model_name = 'roberta-base'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(SAVE_PATH)


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=TEXT_SAVE_PATH,  # mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=TEXT_SAVE_PATH,  # mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=CHK_PATH,  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=200,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)


trainer.train()
trainer.save_model(SAVE_PATH)
