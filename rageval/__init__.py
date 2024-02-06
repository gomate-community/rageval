# from .evaluations import evaluate

try:
    from .version import version as __version__
except ImportError:
    __version__ = "unknown version"

import nltk
nltk.download('punkt')

from . import tasks
from . import metrics
from . import models
from . import utils

# __all__ = ["evaluate", "__version__"]
