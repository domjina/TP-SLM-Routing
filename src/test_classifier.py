from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

save_path = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path)
model.eval()

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

result = clf("The problem from HMMT February is a combinatorics problem requiring some knowledge about number theory")
print(result)
