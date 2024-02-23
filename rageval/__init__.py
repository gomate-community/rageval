try:
    from .version import version as __version__
except ImportError:
    __version__ = "unknown version"

from . import tasks
from . import metrics
from . import models
from . import utils

from .evaluations import evaluate

# __all__ = ["evaluate", "__version__"]
