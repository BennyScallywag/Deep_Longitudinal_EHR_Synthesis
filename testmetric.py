from synthcity.plugins import Plugins
from synthcity.metrics import Metrics
#from synthcity.metrics.eval_metrics import AlphaPrecision, BetaRecall, TSAuthenticity
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


static_data, temporal_data, horizons, outcome = GoogleStocksDataloader().load()
data = TimeSeriesDataLoader(
    temporal_data=temporal_data,
    observation_times=horizons,
    static_data=static_data,
    outcome=outcome,
)

syn_model = Plugins().get("timegan")
syn_model.fit(data)
synthetic_data = syn_model.generate(count=100)

metrics = Metrics.evaluate(
    ["alpha_precision", "beta_recall", "ts_authenticity"], 
    data, 
    synthetic_data
)

print(metrics)
