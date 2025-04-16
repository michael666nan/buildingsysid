"""Building system identification tools."""

# Import key classes and functions to make them available at the package level
from .data.iddata import IDData
from .calculate.pem import pem
from .model_set.grey import predefined as grey
from .model_set.black import canonical as black
from .validation.compare import compare

# Define package version
__version__ = "0.1.0"