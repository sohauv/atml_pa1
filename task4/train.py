"""Train fixed Vanilla and GCSC CIFAR-10 classifiers for Task 4."""

from __future__ import annotations

import argparse, csv, json, random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from task4.data.cifar import build_cifar10
from task4.models import CIFARResNet18


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as file: return yaml.safe_load(file)


def merge(base, override):
    result=dict(base)
    for key,value in override.items(): result[key]=merge(result[key],value) if key in result and isinstance(result[key],dict) and isinstance(value,dict) else value
    return result


def load_config(method):
    return merge(load_yaml("task4/configs/base.yaml"), load_yaml(f"task4/configs/{method}.yaml"))


def loaders(config, root, create_split):
    method=config["method"]
    train, validation, test, manifest=build_cifar10(root, config["data"]["split_manifest"], randaugment=method["randaugment"], create=create_split, seed=config["experiment"]["seed"])
    common={"batch_size":config["data"]["batch_size"],"num_workers":config["data"]["num_workers"],"pin_memory":True}
    return DataLoader(train,shuffle=True,**common), DataLoader(validation,shuffle=False,**common), DataLoader(test,shuffle=False,**common), manifest


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); truth=[]; predictions=[]; losses=[]
    for batch in loader:
        images=batch["image"].to(device); labels=batch["label"].to(device); logits=model(images)
        losses.append(nn.functional.cross_entropy(logits,labels).item()*len(labels)); truth.extend(labels.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
    return {"loss":sum(losses)/len(loader.dataset),"accuracy":accuracy_score(truth,predictions),"macro_f1":f1_score(truth,predictions,average="macro")}


def write_history(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(args):
    config=load_config(args.method); config["data"]["root"]=args.data_root; seed_everything(config["experiment"]["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    train_loader,validation_loader,_,manifest=loaders(config,args.data_root,args.create_split)
    print(f"Train: {len(train_loader.dataset)} | validation: {len(validation_loader.dataset)} | CIFAR-100 is not loaded")
    model=CIFARResNet18(config["model"]["number_of_classes"]).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=config["training"]["learning_rate"],momentum=config["training"]["momentum"],weight_decay=config["training"]["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["training"]["epochs"])
    scaler=torch.amp.GradScaler("cuda",enabled=config["training"]["use_automatic_mixed_precision"] and device.type=="cuda")
    run_name=args.run_name or args.method; checkpoint_path=Path("task4/checkpoints")/f"{run_name}_best.pt"; history_path=Path("task4/results/training")/f"{run_name}_history.csv"
    config_path=Path("task4/results/configs")/f"{run_name}.json"; config_path.parent.mkdir(parents=True,exist_ok=True); config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    best=-1; history=[]
    for epoch in range(config["training"]["epochs"]):
        model.train(); total_loss=0; correct=0; count=0
        for batch in train_loader:
            images=batch["image"].to(device); labels=batch["label"].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=scaler.is_enabled()): logits=model(images); loss=nn.functional.cross_entropy(logits,labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss+=loss.detach().item()*len(labels); correct+=(logits.detach().argmax(1)==labels).sum().item(); count+=len(labels)
        validation=evaluate(model,validation_loader,device); scheduler.step()
        row={"epoch":epoch+1,"learning_rate":optimizer.param_groups[0]["lr"],"train_loss":total_loss/count,"train_accuracy":correct/count,"validation_loss":validation["loss"],"validation_accuracy":validation["accuracy"],"validation_macro_f1":validation["macro_f1"]}; history.append(row); write_history(history,history_path)
        print(f"Epoch {epoch+1:03d} | train loss {row['train_loss']:.4f} | val acc {row['validation_accuracy']:.4f}")
        if validation["accuracy"]>best:
            best=validation["accuracy"]; checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"epoch":epoch+1,"config":config,"class_names":manifest["class_names"],"model_state_dict":model.state_dict(),"validation":validation},checkpoint_path); print(f"Saved {checkpoint_path}")
    print(f"Training complete: {run_name} | best validation accuracy {best:.4f}")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--method",choices=("vanilla","gcsc"),required=True); parser.add_argument("--data-root",required=True); parser.add_argument("--run-name"); parser.add_argument("--create-split",action="store_true"); return parser.parse_args()


if __name__=="__main__": run(parse_args())
