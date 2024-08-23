# flake8: noqa
import os

import torch
from dotmap import DotMap

from pipeline.models.loss import RMSE


def get_config():
    CONF = DotMap()

    # Wandb configuration
    CONF.wandb.project_name = "Seminar ML for Renewable Energy System"
    CONF.wandb.entity_name = "Seminar ML for Renewable Energy System"
    CONF.wandb.sweep_name = "011_time_data_and_weather_forecast_24h_full_transformer"

    # Fixed variables, don't change
    CONF.data.loaded_raw_data = False  # Don't change this

    # Pipeline configuration
    CONF.pipeline.process_raw_data = False
    CONF.pipeline.price_differencing = False
    CONF.pipeline.normalize_data = False
    CONF.pipeline.feature_selection = False
    CONF.pipeline.data_test = False
    CONF.pipeline.do_test_run_training = False
    CONF.pipeline.do_hyperopt = False
    CONF.pipeline.do_final_train = True
    CONF.pipeline.do_test = True
    CONF.pipeline.plot = False  # False to make faster
    CONF.pipeline.inspect = False  # False to make faster

    # Data directory configuration
    CONF.data.data_dir = "/graphics/scratch2/students/nguyenlo/seminar-ml"
    CONF.data.raw_data_dir = os.path.join(CONF.data.data_dir, "raw_data")
    CONF.data.raw_inspection_dir = os.path.join(
        CONF.data.data_dir, "raw_data_inspection"
    )
    CONF.data.preprocessed_data_dir = os.path.join(
        CONF.data.data_dir, "preprocessed_data"
    )
    CONF.data.preprocessed_data_inspection_dir = os.path.join(
        CONF.data.data_dir, "preprocessed_data_inspection"
    )
    os.makedirs(CONF.data.raw_data_dir, exist_ok=True)
    os.makedirs(CONF.data.raw_inspection_dir, exist_ok=True)
    os.makedirs(CONF.data.preprocessed_data_dir, exist_ok=True)
    os.makedirs(CONF.data.preprocessed_data_inspection_dir, exist_ok=True)

    # Data preprocessing configuration
    CONF.data.na_values = "drop_columns"  # drop_rows, drop_columns, fillna
    CONF.data.price_normalization_constant = 10000.0  # euro/MWh

    # Time series prediction configuration
    CONF.model.horizons = [1, 24]

    # Save paths configuration.
    CONF.model.save_path = os.path.join(
        CONF.data.data_dir, "models", CONF.wandb.sweep_name
    )

    # Configurations when doing evaluation.
    CONF.model.final_model_path = os.path.join(
        CONF.data.data_dir,
        "models",
        "011_time_data_and_weather_forecast_24h_full_transformer",
        "best",
        "model.pth",
    )
    CONF.model.best_hyperparameter_path = os.path.join(
        CONF.data.data_dir,
        "models",
        "011_time_data_and_weather_forecast_24h_full_transformer",
        "cgpoolsand1900_22:56:09:970951 20-06-2_654",
        "hyperparameters.pth",
    )

    # Configuration for feature selection
    CONF.feature_selection.all_features = [
        # "Date from",
        # "Date to",
        "prices_Germany/Luxembourg [€/MWh]",
        "prices_Belgium [€/MWh]",
        "prices_Denmark 1 [€/MWh]",
        "prices_Denmark 2 [€/MWh]",
        "prices_France [€/MWh]",
        "prices_Netherlands [€/MWh]",
        "prices_Norway 2 [€/MWh]",
        "prices_Austria [€/MWh]",
        "prices_Sweden 4 [€/MWh]",
        "prices_Switzerland [€/MWh]",
        "prices_Czech Republic [€/MWh]",
        "prices_Italy (North) [€/MWh]",
        "prices_Slovenia [€/MWh]",
        "prices_diff_Germany/Luxembourg [€/MWh]",
        "prices_diff_Belgium [€/MWh]",
        "prices_diff_Denmark 1 [€/MWh]",
        "prices_diff_Denmark 2 [€/MWh]",
        "prices_diff_France [€/MWh]",
        "prices_diff_Netherlands [€/MWh]",
        "prices_diff_Norway 2 [€/MWh]",
        "prices_diff_Austria [€/MWh]",
        "prices_diff_Sweden 4 [€/MWh]",
        "prices_diff_Switzerland [€/MWh]",
        "prices_diff_Czech Republic [€/MWh]",
        "prices_diff_Italy (North) [€/MWh]",
        "prices_diff_Slovenia [€/MWh]",
        "capacity_Biomass [MW]",
        "capacity_Hydro Power [MW]",
        "capacity_Wind Offshore [MW] ",
        "capacity_Wind Onshore [MW]",
        "capacity_Photovoltaic [MW]",
        "capacity_Other Renewable [MW]",
        "capacity_Nuclear Power [MW]",
        "capacity_Lignite [MW]",
        "capacity_Coal [MW]",
        "capacity_Natural Gas [MW]",
        "capacity_Pumped Storage [MW]",
        "capacity_Other Conventional [MW]",
        "supply_Biomass [MW]",
        "supply_Hydro Power [MW]",
        "supply_Wind Offshore [MW] ",
        "supply_Wind Onshore [MW]",
        "supply_Photovoltaic [MW]",
        "supply_Other Renewable [MW]",
        "supply_Nuclear Power [MW]",
        "supply_Lignite [MW]",
        "supply_Coal [MW]",
        "supply_Natural Gas [MW]",
        "supply_Pumped Storage [MW]",
        "supply_Other Conventional [MW]",
        "demand_Total (Grid Load) [MWh]",
        "demand_Residual Load [MWh]",
        "demand_Pumped Storage [MWh]",
        # "weather_forecast_origin",
        "weather_cdir_min",
        "weather_cdir_max",
        "weather_cdir_mean",
        "weather_z_min",
        "weather_z_max",
        "weather_z_mean",
        "weather_msl_min",
        "weather_msl_max",
        "weather_msl_mean",
        "weather_blh_min",
        "weather_blh_max",
        "weather_blh_mean",
        "weather_tcc_min",
        "weather_tcc_max",
        "weather_tcc_mean",
        "weather_u10_min",
        "weather_u10_max",
        "weather_u10_mean",
        "weather_v10_min",
        "weather_v10_max",
        "weather_v10_mean",
        "weather_t2m_min",
        "weather_t2m_max",
        "weather_t2m_mean",
        "weather_ssr_min",
        "weather_ssr_max",
        "weather_ssr_mean",
        "weather_tsr_min",
        "weather_tsr_max",
        "weather_tsr_mean",
        "weather_sund_min",
        "weather_sund_max",
        "weather_sund_mean",
        "weather_tp_min",
        "weather_tp_max",
        "weather_tp_mean",
        "weather_fsr_min",
        "weather_fsr_max",
        "weather_fsr_mean",
        "weather_u100_min",
        "weather_u100_max",
        "weather_u100_mean",
        "weather_v100_min",
        "weather_v100_max",
        "weather_v100_mean",
        # "train",
        # "val",
        # "test",
    ]

    CONF.model.weather_features = [
        "weather_cdir_min",
        "weather_cdir_max",
        "weather_cdir_mean",
        "weather_z_min",
        "weather_z_max",
        "weather_z_mean",
        "weather_msl_min",
        "weather_msl_max",
        "weather_msl_mean",
        "weather_blh_min",
        "weather_blh_max",
        "weather_blh_mean",
        "weather_tcc_min",
        "weather_tcc_max",
        "weather_tcc_mean",
        "weather_u10_min",
        "weather_u10_max",
        "weather_u10_mean",
        "weather_v10_min",
        "weather_v10_max",
        "weather_v10_mean",
        "weather_t2m_min",
        "weather_t2m_max",
        "weather_t2m_mean",
        "weather_ssr_min",
        "weather_ssr_max",
        "weather_ssr_mean",
        "weather_tsr_min",
        "weather_tsr_max",
        "weather_tsr_mean",
        "weather_sund_min",
        "weather_sund_max",
        "weather_sund_mean",
        "weather_tp_min",
        "weather_tp_max",
        "weather_tp_mean",
        "weather_fsr_min",
        "weather_fsr_max",
        "weather_fsr_mean",
        "weather_u100_min",
        "weather_u100_max",
        "weather_u100_mean",
        "weather_v100_min",
        "weather_v100_max",
        "weather_v100_mean",
    ]

    # Time series hyper parameters
    CONF.model.features = [
        # "prices_Germany/Luxembourg [€/MWh]",
        # "prices_Belgium [€/MWh]",
        # "prices_Denmark 1 [€/MWh]",
        # "prices_Denmark 2 [€/MWh]",
        # "prices_France [€/MWh]",
        # "prices_Netherlands [€/MWh]",
        # "prices_Norway 2 [€/MWh]",
        # "prices_Austria [€/MWh]",
        # "prices_Sweden 4 [€/MWh]",
        # "prices_Switzerland [€/MWh]",
        # "prices_Czech Republic [€/MWh]",
        # "prices_Italy (North) [€/MWh]",
        # "prices_Slovenia [€/MWh]",
        "prices_diff_Germany/Luxembourg [€/MWh]",
        "prices_diff_Belgium [€/MWh]",
        "prices_diff_Denmark 1 [€/MWh]",
        "prices_diff_Denmark 2 [€/MWh]",
        "prices_diff_France [€/MWh]",
        "prices_diff_Netherlands [€/MWh]",
        "prices_diff_Norway 2 [€/MWh]",
        "prices_diff_Austria [€/MWh]",
        "prices_diff_Sweden 4 [€/MWh]",
        "prices_diff_Switzerland [€/MWh]",
        "prices_diff_Czech Republic [€/MWh]",
        "prices_diff_Italy (North) [€/MWh]",
        "prices_diff_Slovenia [€/MWh]",
        "capacity_Biomass [MW]",
        "capacity_Hydro Power [MW]",
        "capacity_Wind Offshore [MW] ",
        "capacity_Wind Onshore [MW]",
        "capacity_Photovoltaic [MW]",
        "capacity_Other Renewable [MW]",
        "capacity_Nuclear Power [MW]",
        "capacity_Lignite [MW]",
        "capacity_Coal [MW]",
        "capacity_Natural Gas [MW]",
        "capacity_Pumped Storage [MW]",
        "capacity_Other Conventional [MW]",
        "supply_Biomass [MW]",
        "supply_Hydro Power [MW]",
        # "supply_Wind Offshore [MW] ",
        # "supply_Wind Onshore [MW]",
        # "supply_Photovoltaic [MW]",
        "supply_Other Renewable [MW]",
        "supply_Nuclear Power [MW]",
        "supply_Lignite [MW]",
        "supply_Coal [MW]",
        "supply_Natural Gas [MW]",
        "supply_Pumped Storage [MW]",
        "supply_Other Conventional [MW]",
        "demand_Total (Grid Load) [MWh]",
        "demand_Residual Load [MWh]",
        "demand_Pumped Storage [MWh]",
        # "weather_forecast_origin",
        "weather_cdir_min",
        "weather_cdir_max",
        "weather_cdir_mean",
        "weather_z_min",
        "weather_z_max",
        "weather_z_mean",
        "weather_msl_min",
        "weather_msl_max",
        "weather_msl_mean",
        "weather_blh_min",
        "weather_blh_max",
        "weather_blh_mean",
        "weather_tcc_min",
        "weather_tcc_max",
        "weather_tcc_mean",
        "weather_u10_min",
        "weather_u10_max",
        "weather_u10_mean",
        "weather_v10_min",
        "weather_v10_max",
        "weather_v10_mean",
        "weather_t2m_min",
        "weather_t2m_max",
        "weather_t2m_mean",
        "weather_ssr_min",
        "weather_ssr_max",
        "weather_ssr_mean",
        "weather_tsr_min",
        "weather_tsr_max",
        "weather_tsr_mean",
        "weather_sund_min",
        "weather_sund_max",
        "weather_sund_mean",
        "weather_tp_min",
        "weather_tp_max",
        "weather_tp_mean",
        "weather_fsr_min",
        "weather_fsr_max",
        "weather_fsr_mean",
        "weather_u100_min",
        "weather_u100_max",
        "weather_u100_mean",
        "weather_v100_min",
        "weather_v100_max",
        "weather_v100_mean",
    ]
    CONF.model.targets = [
        "supply_Wind Offshore [MW] ",
        "supply_Wind Onshore [MW]",
        "supply_Photovoltaic [MW]",
    ]

    # Transformer's architecture's fixed hyperparameters
    CONF.model.num_targets = len(CONF.model.targets)
    CONF.model.num_features = (
        len(CONF.model.features) + len(CONF.model.targets) + 2
    )  # 2 for hour of day and day of year + forecasts in future
    CONF.model.decoder_num_features = len(CONF.model.weather_features)

    # Transformer's architecture's tunable hyperparameters
    CONF.model.architecture = "HorizonTargetTransformer"
    CONF.model.num_layers = 1
    CONF.model.num_heads = 1
    CONF.model.dropout = 0.1
    CONF.model.lag = 12
    CONF.model.dim_feedforward_factor = 0.5

    # Training tunable hyperparameters
    CONF.train.batch_size = 256
    CONF.train.lr = 0.001
    CONF.train.min_lr = 0.00001
    CONF.train.clip_grad = 1.0
    CONF.train.optimizer = torch.optim.AdamW

    # Fixed hyper parameters
    CONF.train.loss = RMSE
    CONF.train.patience = 20
    CONF.train.epochs = 100
    CONF.train.hyperparameters_iters = 10000
    return CONF


CONF = get_config()
