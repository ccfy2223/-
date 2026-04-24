import torch


def main() -> None:
    print("torch", torch.__version__)
    print("cuda_version", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))

    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    print("autogluon_timeseries_import_ok", TimeSeriesDataFrame.__name__, TimeSeriesPredictor.__name__)


if __name__ == "__main__":
    main()
