import os
import csv


__all__ = ["evaluate", "evaluate_dataset"]


def evaluate(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    metrics,
    custom_header=None,
):

    header = []
    results = []
    if "train" in args.split:
        header += ["Train MAE", "Train RMSE"]
        results += evaluate_dataset(metrics, model, train_loader, device)
    if "validation" in args.split:
        header += ["Val MAE", "Val RMSE"]
        results += evaluate_dataset(metrics, model, val_loader, device)

    if "test" in args.split:
        header += ["Test MAE", "Test RMSE"]
        results += evaluate_dataset(metrics, model, test_loader, device)

    if custom_header:
        header = custom_header

    eval_file = os.path.join(args.modelpath, "evaluation.txt")
    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(results)


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [metric.aggregate() for metric in metrics]
    return results
