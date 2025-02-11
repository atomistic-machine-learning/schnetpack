from src.schnetpack.data.datamodule import AtomsDataModule


datamodule = AtomsDataModule(
    datapath="mp_nextgen.db",
    batch_size=10,
    num_train=1004,
    num_val=20,
    num_test=10,
    split_file="split.npz"
)

datamodule.setup()
