try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

name = "Thor"
package_name = "thor"
try:
    __version__ = importlib_metadata.version(package_name)
except importlib_metadata.PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.0.0"

import warnings
warnings.filterwarnings("ignore")

import logging
import sys


def _in_notebook() -> bool:
    """Detect classic and JupyterLab notebooks without importing IPython when unavailable."""
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:  # pragma: no cover - IPython not installed or misconfigured
        return False

    shell = get_ipython()
    if shell is None:
        return False

    shell_name = shell.__class__.__name__
    return shell_name.endswith("InteractiveShell") and "ZMQ" in shell_name


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# nice logging outputs
from rich.console import Console
from rich.logging import RichHandler

_running_in_notebook = _in_notebook()
_has_tty = sys.stdout.isatty()
console = Console(
    width=150,
    force_terminal=_running_in_notebook,
    force_jupyter=_running_in_notebook,
    color_system="auto" if (_running_in_notebook or _has_tty) else None,
)

ch = RichHandler(show_path=False, console=console, show_time=True)
formatter = logging.Formatter("Thor: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False


from . import pl, utils, analy, pp, VAE
from .finest import fineST
