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

    loaders = dict(train=train_loader, validation=val_loader, test=test_loader)
    for datasplit in args.split:
        header += ["{} MAE".format(datasplit), "{} RMSE".format(datasplit)]
        derivative = model.output_modules[0].derivative
        if derivative is not None:
            header += [
                "{} MAE ({})".format(datasplit, derivative),
                "{} RMSE ({})".format(datasplit, derivative),
            ]
        results += evaluate_dataset(metrics, model, loaders[datasplit], device)

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
