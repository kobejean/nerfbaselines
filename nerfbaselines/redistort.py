import struct
import sys
import json
import hashlib
from functools import partial
import time
import os
import math
import logging
from pathlib import Path
import typing
from typing import Callable, Optional, Union, Type, List, Any, Dict, Tuple, cast
from typing import TYPE_CHECKING
from tqdm import tqdm
import numpy as np
from PIL import Image
import click
from .io import open_any_directory
from .datasets import load_dataset, Dataset
from .utils import Indices, setup_logging, partialclass, image_to_srgb, visualize_depth, handle_cli_error
from .utils import remap_error, convert_image_dtype, get_resources_utilization_info, assert_not_none
from .types import Method, CurrentProgress, ColorSpace, Literal, FrozenSet
from .render import render_all_images, with_supported_camera_models
from .upload_results import prepare_results_for_upload
from . import __version__
from . import registry
from . import evaluate

if TYPE_CHECKING:
    import wandb.sdk.wandb_run
    from .tensorboard import TensorboardWriter

    _wandb_type = type(wandb)



def make_grid(*images: np.ndarray, ncol=None, padding=2, max_width=1920, background=1.0):
    if ncol is None:
        ncol = len(images)
    dtype = images[0].dtype
    background = convert_image_dtype(
        np.array(background, dtype=np.float32 if isinstance(background, float) else np.uint8),
        dtype).item()
    nrow = int(math.ceil(len(images) / ncol))
    scale_factor = 1
    height, width = tuple(map(int, np.max([x.shape[:2] for x in images], axis=0).tolist()))
    if max_width is not None:
        scale_factor = min(1, (max_width - padding * (ncol - 1)) / (ncol * width))
        height = int(height * scale_factor)
        width = int(width * scale_factor)

    def interpolate(image) -> np.ndarray:
        img = Image.fromarray(image)
        img_width, img_height = img.size
        aspect = img_width / img_height
        img_width = int(min(width, aspect * height))
        img_height = int(img_width / aspect)
        img = img.resize((img_width, img_height))
        return np.array(img)

    images = tuple(map(interpolate, images))
    grid: np.ndarray = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[-1]),
        dtype=dtype,
    )
    grid.fill(background)
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        h, w = image.shape[:2]
        offx = x * (width + padding) + (width - w) // 2
        offy = y * (height + padding) + (height - h) // 2
        grid[offy : offy + h, 
             offx : offx + w] = image
    return grid


def compute_exponential_gamma(num_iters: int, initial_lr: float, final_lr: float) -> float:
    gamma = (math.log(final_lr) - math.log(initial_lr)) / num_iters
    return math.exp(gamma)


def method_get_resources_utilization_info(method):
    if hasattr(method, "call"):
        return method.call(f"{get_resources_utilization_info.__module__}.{get_resources_utilization_info.__name__}")
    return get_resources_utilization_info()


Visualization = Literal["none", "wandb", "tensorboard"]


class Redistorter:
    def __init__(
        self,
        *,
        train_dataset: Union[str, Path, Callable[[], Dataset]],
        test_dataset: Union[None, str, Path, Callable[[], Dataset]] = None,
        method: Type[Method],
        output: Path = Path("."),
        num_iterations: Optional[int] = None,
        save_iters: Indices = Indices.every_iters(10_000, zero=True),
        eval_few_iters: Indices = Indices.every_iters(2_000),
        eval_all_iters: Indices = Indices([-1]),
        loggers: FrozenSet[Visualization] = frozenset(),
        color_space: Optional[ColorSpace] = None,
        run_extra_metrics: bool = False,
        method_name: Optional[str] = None,
        generate_output_artifact: Optional[bool] = None,
        checkpoint: Union[str, Path, None] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.method_name = method_name or method.__name__
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.method = method(**({"checkpoint": checkpoint} if checkpoint is not None else {}))
        method_info = self.method.get_info()
        if isinstance(train_dataset, (Path, str)):
            if test_dataset is None:
                test_dataset = train_dataset
            train_dataset_path = train_dataset
            train_dataset = partial(load_dataset, train_dataset, split="train", features=method_info.required_features)

            # Allow direct access to the stored images if needed for docker remote
            if hasattr(self.method, "mounts"):
                typing.cast(Any, self.method).mounts.append((str(train_dataset_path), str(train_dataset_path)))
        if isinstance(test_dataset, (Path, str)):
            test_dataset = partial(load_dataset, test_dataset, split="test", features=method_info.required_features)
        assert test_dataset is not None, "test dataset must be specified"
        self._train_dataset_fn: Callable[[], Dataset] = train_dataset
        self._test_dataset_fn: Callable[[], Dataset] = test_dataset
        self.test_dataset: Optional[Dataset] = None

        self.step = method_info.loaded_step or 0
        self.output = output
        self.num_iterations = num_iterations
        self.save_iters = save_iters

        self.eval_few_iters = eval_few_iters
        self.eval_all_iters = eval_all_iters
        self.loggers = loggers
        self.run_extra_metrics = run_extra_metrics
        self.generate_output_artifact = generate_output_artifact
        self.config_overrides = config_overrides
        self._wandb_run: Union["wandb.sdk.wandb_run.Run", None] = None
        self._tensorboard_writer: Optional['TensorboardWriter'] = None
        self._average_image_size = None
        self._color_space = color_space
        self._expected_scene_scale = None
        self._method_info = method_info
        self._total_train_time = 0
        self._resources_utilization_info = None
        self._dataset_background_color = None
        self._train_dataset_for_eval = None

        # Restore checkpoint if specified
        if self.checkpoint is not None:
            with open(self.checkpoint / "nb-info.json", mode="r", encoding="utf8") as f:
                info = json.load(f)
                self._total_train_time = info["total_train_time"]
                self._resources_utilization_info = info["resources_utilization"]


    @remap_error
    def setup_data(self):
        logging.info("loading eval dataset")
        self.test_dataset = self._test_dataset_fn()
        method_info = self.method.get_info()

        self.test_dataset.redistort_features(method_info.required_features, method_info.supported_camera_models)
        assert self.test_dataset.cameras.image_sizes is not None, "image sizes must be specified"
        


class IndicesClickType(click.ParamType):
    name = "indices"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, Indices):
            return value
        if ":" in value:
            parts = [int(x) if x else None for x in value.split(":")]
            assert len(parts) <= 3, "too many parts in slice"
            return Indices(slice(*parts))
        return Indices([int(x) for x in value.split(",")])


class SetParamOptionType(click.ParamType):
    name = "key-value"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, tuple):
            return value
        if "=" not in value:
            self.fail(f"expected key=value pair, got {value}", param, ctx)
        k, v = value.split("=", 1)
        return k, v


@click.command("redistort")
@click.option("--method", type=click.Choice(sorted(registry.supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--vis", type=click.Choice(["none", "wandb", "tensorboard", "wandb+tensorboard"]), default="tensorboard", help="Logger to use. Defaults to tensorboard.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--eval-few-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate on few images")
@click.option("--eval-all-iters", type=IndicesClickType(), default=Indices([-1]), help="When to evaluate all images")
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--disable-extra-metrics", help="Disable extra metrics which need additional dependencies.", is_flag=True)
@click.option("--disable-output-artifact", "generate_output_artifact", help="Disable producing output artifact containing final model and predictions.", default=None, flag_value=False, is_flag=True)
@click.option("--force-output-artifact", "generate_output_artifact", help="Force producing output artifact containing final model and predictions.", default=None, flag_value=True, is_flag=True)
@click.option("--num-iterations", type=int, help="Number of redistorting iterations.", default=None)
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@handle_cli_error
def redistort_command(
    method,
    checkpoint,
    data,
    output,
    verbose,
    backend,
    eval_few_iters,
    eval_all_iters,
    num_iterations=None,
    disable_extra_metrics=False,
    generate_output_artifact=None,
    vis="none",
    config_overrides=None,
):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    if not disable_extra_metrics:
        try:
            evaluate.test_extra_metrics()
        except ImportError as exc:
            logging.error(exc)
            logging.error("Extra metrics are not available and will be disabled. Please install the required dependencies by running `pip install nerfbaselines[extras]`.")
            disable_extra_metrics = True

    if method is None and checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    def _redistort(checkpoint_path=None):
        with open_any_directory(output, "w") as output_path:
            # Make paths absolute, and change working directory to output
            _data = data
            if "://" not in _data:
                _data = os.path.abspath(_data)
            os.chdir(str(output_path))

            method_spec = registry.get(method)
            _method, _backend = method_spec.build(backend=backend, checkpoint=Path(os.path.abspath(checkpoint_path)) if checkpoint_path else None)
            logging.info(f"Using method: {method}, backend: {_backend}")

            # Enable direct memory access to images and if supported by the backend
            if _backend in {"docker", "apptainer"} and "://" not in _data:
                _method = partialclass(_method, mounts=[(_data, _data)])
            if hasattr(_method, "install"):
                _method.install()

            loggers: FrozenSet[Visualization]
            if vis == "wandb":
                loggers = frozenset(("wandb",))
            elif vis == "tensorboard":
                loggers = frozenset(("tensorboard",))
            elif vis in {"wandb+tensorboard", "tensorboard+wandb"}:
                loggers = frozenset(("wandb", "tensorboard"))
            elif vis == "none":
                loggers = frozenset()
            else:
                raise ValueError(f"unknown visualization tool {vis}")

            redistorter = Redistorter(
                redistort_dataset=_data,
                output=Path(output),
                method=_method,
                eval_all_iters=eval_all_iters,
                eval_few_iters=eval_few_iters,
                loggers=frozenset(loggers),
                num_iterations=num_iterations,
                run_extra_metrics=not disable_extra_metrics,
                generate_output_artifact=generate_output_artifact,
                method_name=method,
                config_overrides=config_overrides,
            )
            try:
                redistorter.setup_data()
            finally:
                pass

    if checkpoint is not None:
        with open_any_directory(checkpoint) as checkpoint_path:
            with open(os.path.join(checkpoint_path, "nb-info.json"), "r", encoding="utf8") as f:
                info = json.load(f)
            if method is not None and method != info["method"]:
                logging.error(f"Argument --method={method} is in conflict with the checkpoint's method {info['method']}.")
                sys.exit(1)
            method = info["method"]
            _redistort(checkpoint_path)
    else:
        _redistort(None)


if __name__ == "__main__":
    redistort_command()  # pylint: disable=no-value-for-parameter
