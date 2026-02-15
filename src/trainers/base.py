import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

def train(model, dataloader, criterion, optimizer, edge_index, binary, device):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch, los_batch in tqdm(dataloader, desc="train_process", leave=True):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)

        optimizer.zero_grad()

        logits = model(
            x_batch,
            los_batch,
            edge_index,
            device=device,
        )
        if binary: 
            logits = logits.squeeze(1)
            loss = criterion(logits, y_batch.float())
        else:
            loss = criterion(logits, y_batch.long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(
    model,
    val_dataloader,
    criterion,
    decision_threshold,
    device,
    binary,
    edge_index,
    num_classes: int | None = None,   # multiclass일 때만 필요 (없으면 자동 추정 시도)
):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_scores = []  # binary: (N,), multiclass: (N, K)

    with torch.no_grad():
        for x_batch, y_batch, los_batch in tqdm(val_dataloader, desc="eval_process", leave=True):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            los_batch = los_batch.to(device)

            logits = model(x_batch, 
                           los_batch, 
                           edge_index, 
                           device=device)

            if binary:
                # logits: [B, 1] or [B]
                if logits.ndim == 2 and logits.size(1) == 1:
                    logits_1d = logits.squeeze(1)   # [B]
                else:
                    logits_1d = logits              # [B]

                loss = criterion(logits_1d, y_batch.float())

                scores = torch.sigmoid(logits_1d)  # [B]
                predicted = (scores >= decision_threshold).long()  # [B]

                all_scores.append(scores.detach().cpu().numpy())  # (B,)

            else:
                # multiclass logits: [B, K]
                # y_batch must be int class indices: [B]
                loss = criterion(logits, y_batch.long())

                probs = F.softmax(logits, dim=1)         # [B, K]
                predicted = torch.argmax(probs, dim=1)   # [B]

                all_scores.append(probs.detach().cpu().numpy())   # (B, K)

            running_loss += loss.item() * x_batch.size(0)

            all_targets.append(y_batch.detach().cpu().numpy())
            all_predictions.append(predicted.detach().cpu().numpy())

            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    all_targets = np.concatenate(all_targets)         # (N,)
    all_predictions = np.concatenate(all_predictions) # (N,)
    all_scores = np.concatenate(all_scores)           # binary: (N,), multiclass: (N, K)

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_accuracy = total_correct / total_samples

    # precision/recall/f1
    # binary도 macro 써도 되긴 하지만, binary면 average='binary'가 더 직관적일 때가 많음
    if binary:
        epoch_precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
        epoch_recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
        epoch_f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
    else:
        epoch_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        epoch_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    # AUC
    try:
        if binary:
            epoch_auc = roc_auc_score(all_targets, all_scores)  # all_scores: (N,)
        else:
            # num_classes가 없으면 scores shape로 추정
            K = num_classes if num_classes is not None else all_scores.shape[1]
            # multiclass AUC: (N,K) probs + (N,) labels
            epoch_auc = roc_auc_score(all_targets, all_scores, multi_class='ovr', average='macro')
    except ValueError:
        print("Warning: AUC score could not be calculated (maybe missing classes in targets).")
        epoch_auc = 0.0

    # label counts 출력 (전체 누적 기준)
    if binary:
        print("Valid preds label counts:", np.bincount(all_predictions.astype(int), minlength=2))
        print("Valid true label counts:", np.bincount(all_targets.astype(int), minlength=2))
    else:
        K = num_classes if num_classes is not None else int(all_scores.shape[1])
        print("Valid preds label counts:", np.bincount(all_predictions.astype(int), minlength=K))
        print("Valid true label counts:", np.bincount(all_targets.astype(int), minlength=K))

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_auc


'''def evaluate(model, val_dataloader, criterion, decision_threshold, device, edge_index):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for x_batch, y_batch, los_batch in tqdm(val_dataloader, desc="eval_process", leave=True):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            los_batch = los_batch.to(device)

            logits = model(
                x_batch,
                los_batch,
                edge_index,
            )

            logits = logits.squeeze(1)
            loss = criterion(logits, y_batch.float())

            with torch.no_grad():
                scores = torch.sigmoid(logits)            # [B]
                predicted = (scores >= decision_threshold).long()

            running_loss += loss.item() * x_batch.size(0)

            all_targets.append(y_batch.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_scores.append(scores.cpu().numpy()) # AUC 계산을 위해 확률(Scores) 저장
            
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_scores = np.concatenate(all_scores)

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_accuracy = total_correct / total_samples

    try:
        # Since this is binary classification, all_scores must be (N,) not (N, 1).
        # squeeze(1) ensures the correct dimensionality.
        epoch_auc = roc_auc_score(all_targets, all_scores) 
    except ValueError:
        print("Warning: AUC score could not be calculated.")
        epoch_auc = 0.0

    epoch_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    print("Valid preds label counts:", torch.bincount(preds))
    print("Valid true label counts:", torch.bincount(y_batch))


    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_auc'''


def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename, map_location=None):
    """
    저장된 체크포인트(.pth)를 불러와서 
    model, optimizer, scheduler 상태를 복구합니다.

    Parameters:
        model (nn.Module): 모델 객체
        optimizer (torch.optim.Optimizer): 옵티마이저 객체
        scheduler: 스케줄러 객체
        filename (str): 저장된 체크포인트 경로
        map_location: CPU로 로드하고 싶으면 'cpu' 또는 torch.device('cpu')

    Returns:
        start_epoch (int): 다음 훈련을 시작할 epoch 번호
        best_loss (float): 저장된 최소 validation loss
    """
    checkpoint = torch.load(filename, map_location=map_location)

    # --- Load states ---
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    
    best_loss = checkpoint.get('best_loss', None)
    return start_epoch, best_loss

def run_train_loop(
    model,
    edge_index,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    scheduler,
    early_stopper,
    device,
    binary,
    logger=None,
    start_epoch: int = 1,
    **kwargs
):
    EPOCHS = kwargs["epochs"]
    decision_threshold = kwargs["decision_threshold"]

    last_epoch = start_epoch - 1  # 루프가 0번 돌 때 대비

    for epoch in tqdm(range(start_epoch, EPOCHS + 1)):
        last_epoch = epoch

        train_loss = train(model, train_dataloader, criterion, optimizer, edge_index, binary, device)

        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate(
            model, val_dataloader, criterion, decision_threshold, device, binary, edge_index
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "lr": float(current_lr),
            "train_loss": float(train_loss),
            "valid_loss": float(val_loss),
            "valid_acc": float(val_accuracy),
            "valid_precision": float(val_precision),
            "valid_recall": float(val_recall),
            "valid_f1": float(val_f1),
            "valid_auc": float(val_auc),
        }

        if logger is not None:
            logger.log_metrics(epoch, metrics)
            logger.maybe_save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                extra=None,  # edge_index는 일단 저장하지 않기 추천
            )
            if logger.best_epoch == epoch:
                print(f"  ✅ New best saved: valid_loss={val_loss:.4f}")

        print(f"\n[Epoch {epoch}/{EPOCHS}]")
        print(f"  [Train] LR: {current_lr:.6f} | Loss: {train_loss:.4f}")
        print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        should_stop = early_stopper(val_loss)
        if should_stop:
            print("\n--- Early Stopping activated. Learning terminated. ---")
            break

    print("\n--- Training Finished ---")

    with torch.no_grad():
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate(
            model, test_dataloader, criterion, decision_threshold, device, binary, edge_index
        )

    print(f"\n[Test] Loss: {test_loss:.4f} | Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

    if logger is not None:
        logger.log_metrics(last_epoch, {
            "split": "test",
            "test_loss": float(test_loss),
            "test_acc": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "test_auc": float(test_auc),
        })

    result_dict = {
        "split": "test",
        "test_loss": float(test_loss),
        "test_acc": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
    }

    return result_dict
