import inspect
import logging
import pickle
import re
from datetime import date
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

logger = logging.getLogger(__name__)


def get_hash(object_list: List):
    """Hash a list of objects."""
    str_params = ""
    for obj in object_list:
        # regex filtering of pesky RandomState objects
        str_params += re.sub(r"'random[^{]*", "", str(obj))
    hashed_repr = sha256((str_params).encode()).hexdigest()
    return hashed_repr


def get_hashable_repr(func, args, kwargs):
    """Parse arguments into hashable types."""
    hashable_items = [kwargs]
    posargs = args
    if str(inspect.signature(func)).startswith("(self,"):
        # hash instance string representation
        hashable_items.append(str(args[0]))
        posargs = args[1:]
    for arg in posargs:
        if isinstance(arg, pd.DataFrame):
            hashable_items.append(pd.util.hash_pandas_object(arg).sum())
        else:
            hashable_items.append(arg)
    return hashable_items


def use_cache(cache_opts: Dict, configurations: Dict = None):
    """Cache function results.

    Parameters
    ----------
    cache_opts : Dict
        cache configurations

    configurations : Dict, optional
        configurations that define the object's hash, by default None
    """

    def use_cache_decorator(function):
        """Function result caching wrapper."""

        @wraps(function)
        def wrapper(*args, **kwargs):
            hashable_args = get_hashable_repr(function, args, kwargs)
            run_hash = get_hash([configurations, hashable_args])
            filepath = Path(cache_opts["cache_dir"]) / f"{date.today()}-{run_hash}.pkl"
            if cache_opts["active"] and filepath.exists():
                result = from_pickle(filepath)
                logger.info("Using cached object: %s.", filepath)  # noqa
            else:
                result = function(*args, **kwargs)
                if cache_opts["active"]:
                    make_dirs(cache_opts["cache_dir"])
                    to_pickle(filepath, result)
                    logger.info("Cached object to: %s.", filepath)
            return result

        return wrapper

    return use_cache_decorator


def to_pickle(filepath: Union[str, Path], obj: Any):
    """Pickle object."""
    with open(filepath, "wb") as handle:
        pickle.dump(
            obj,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def from_pickle(filepath: Union[str, Path]):
    """Load pickled object."""
    with open(filepath, "rb") as handle:
        result = pickle.load(handle)
    return result


def make_dirs(dir_path):
    """Make directory and return it."""
    Path.mkdir(Path(dir_path), exist_ok=True, parents=True)
    return dir_path
