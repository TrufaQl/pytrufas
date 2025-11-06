import os
import torch
import numpy
import pandas
import datasets
import evaluate
import tensorflow
import ultralytics
import transformers

import boyle as by
# ========== ========== ========== ========== ========== #
def downloadDataSet(name, savePath):
    dataset = datasets.load_dataset(name)
    dataset.save_to_disk(savePath)
    return dataset
def loadDataSet(dataname, loadPath):
    try:
        dataset = loadDataSet(loadPath)
    except:
        dataset = downloadDataSet(dataname, loadPath)
    return dataset

def loadYOLOv8(rootPath, name):
    modelPath = rootPath / "Model" / name
    version = by.localFile(modelPath, "--- nuevo ---")
    if version == "--- nuevo ---":
        yolo = createYOLOv8(rootPath, name)
    else:
        yoloPath = modelPath / version / "weights" / "best.pt"
        yolo = ultralytics.YOLO(yoloPath)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        yolo.to(device)
        
    return yolo
def createYOLOv8(rootPath, name):
    dataPath = rootPath / "Train" / name / "info.yaml"
    savePath = rootPath / "Model" / name

    sizes = ["nano", "small", "medium", "large"] 
    imgszs = ["192", "320", "512", "640", "1024"]

    size = by.customParameter(sizes)[0]
    imgsz = by.customParameter(imgszs)
    batch = by.customValue("batch", 2, 256)
    epochs = by.customValue("epochs", 1, 100)

    yolo = ultralytics.YOLO(f"yolov8{size}.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo.to(device)

    yolo.train(
        data=dataPath,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=savePath,
        name=f"yolov8{size}{imgsz}",
        save_period=-1,
        verbose=False
        )
    return yolo
def inferenceYOLOv8(yolo, imgsPaths):
    imgs = []
    boxes = []
    output = yolo.predict(source=imgsPaths)
    for result in output:
        imgs.append(result.orig_img)
        boxes.append(result.boxes)
    return imgs, boxes
class YOLOv8:
    def __init__(self, rootPath, name):
        self.yolo = loadYOLOv8(rootPath, name)
        return
    def Inference(self, imgsPaths):
        return inferenceYOLOv8(self.yolo, imgsPaths)

def loadTransformer(rootPath, name, version, dataname):
    modelPath = rootPath / "Model" / name
    os.makedirs(modelPath, exist_ok=True)
    local = by.localFile(modelPath, "--- nuevo ---")
    if local == "--- nuevo ---":
        transformer, tokenizer, device = createTransformer(rootPath, name, version, dataname)
        return 
    else:
        optimusPath = modelPath / version
        tokenizer = transformers.AutoTokenizer.from_pretrained(optimusPath)
        transformer = transformers.AutoModelForSequenceClassification.from_pretrained(optimusPath)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transformer.to(device)
        transformer.eval()
    
    return transformer, tokenizer, device
def createTransformer(rootPath, name, version, dataname):
    dataPath = rootPath / "Train" / name
    modelPath = rootPath / "Model" / name
    
    dataset = loadDataSet(dataname, dataPath)
    tokenizer = transformers.AutoTokenizer.from_pretrained(version)
    labels = dataset["train"].features["label"].names

    id2label = { str(i): label for i, label in enumerate(labels) }
    label2id = { label: str(i) for i, label in enumerate(labels) }

    transformer = transformers.AutoModelForSequenceClassification.from_pretrained(
        version, num_labels=len(labels), id2label=id2label, label2id=label2id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer.to(device)

    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True),
        batched=True)
    
    batch = by.customValue("batch", 2, 64)
    epochs = by.customValue("epochs", 1, 30)

    training_args = transformers.TrainingArguments(
        output_dir=modelPath,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        metric_for_best_model="loss",
        load_best_model_at_end = True,
        greater_is_better=False,
        report_to="none")
    
    trainer = transformers.Trainer(
        model=transformer,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=transformers.DataCollatorWithPadding(tokenizer=tokenizer))
    
    trainer.train()

    bestPath = modelPath / "best"
    trainer.save_model(bestPath)

    return transformer, tokenizer, device
def inferenceTransformer(model, tokenizer, device, input, output=False):
    inputs = tokenizer(input, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction_id = torch.argmax(logits, dim=-1).item()
    probabilities = torch.softmax(logits, dim=-1)
    max_prob_tensor, prob_prediction_id_tensor = torch.max(probabilities, dim=-1)
    max_probability = max_prob_tensor.item()
    predicted_label = model.config.id2label[prediction_id]

    if output:
        print("-" * 20)
        print(f"Texto: '{input}'")
        print(f"Predicción (ID): {prediction_id}")
        print(f"Predicción (Etiqueta): {predicted_label}")
        print(f"Probabilidad/Confianza máxima: {max_probability * 100:.2f}%")

    return predicted_label
class Transformer:
    def __init__(self, rootPath, name, version, dataname):
        self.model, self.tokenizer, self.device = loadTransformer(rootPath, name, version, dataname)
        return
    def Inference(self, input):
        return inferenceTransformer(self.model, self.tokenizer, self.device, input)