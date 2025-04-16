from buildingsysid.data.iddata_base import IDData

from buildingsysid.data.iddata_resample import IDDataResample
from buildingsysid.data.iddata_split import IDDataSplit
from buildingsysid.data.iddata_timeseriesplot import IDDataTimeSeriesPlot
from buildingsysid.data.iddata_correlation import IDDataCorrelation
from buildingsysid.data.iddata_normalize import IDDataNormalize

class IDData(IDData, IDDataResample, IDDataSplit, IDDataTimeSeriesPlot, IDDataCorrelation, IDDataNormalize):
    """
    Initialize IDData object for system identification data.
    
    Args:
        y (numpy.ndarray): Output data with shape (num_outputs, num_samples) or (num_samples,)
        u (numpy.ndarray): Input data with shape (num_inputs, num_samples) or (num_samples,)
        samplingTime (float): Sampling time in seconds
        timestamps (array-like): Timestamps for the data points
        y_names (list, optional): Names for each output channel
        u_names (list, optional): Names for each input channel
        y_units (list, optional): Units for each output channel
        u_units (list, optional): Units for each input channel
    """
    pass