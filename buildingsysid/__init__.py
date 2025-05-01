"""Building system identification tools."""

# Import subpackages
from . import calculate
from . import criterion_of_fit
from . import data
from . import model_set
from . import validation
from . import utils


# Import key classes and functions to make them available at package level
from .data.iddata import IDData
from .utils.statespace import StateSpace



# from .validation.compare import compare
# from .validation.step_response import discrete_step_response

# # Calculation
# from .calculate.optimization_problem import OptimizationProblem
# from .calculate.solvers.least_squares_solver import LeastSquaresSolver
# from buildingsysid.criterion_of_fit.objective_functions import StandardObjective

# # Import key classes and functions to make them available at the package level

# from .calculate.pem import pem
# from .model_set.grey import predefined as grey
# from .model_set.black import canonical as black




# Define package version
__version__ = "0.1.4"