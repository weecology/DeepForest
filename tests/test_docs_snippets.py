import re
from pathlib import Path
import types
import sys

# -----------------------------------------------------------------------------
# Safe import of optional heavy dependencies or create lightweight stubs
# -----------------------------------------------------------------------------


def _optional_import(name: str):
    """Import a module if available, else create & register an empty stub."""
    if name in sys.modules:
        return sys.modules[name]

    try:
        module = __import__(name)
    except ModuleNotFoundError:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


# Prepare minimal versions of common scientific packages so that documentation
# snippets that `import` them do not fail inside the test runner.  Only a tiny
# subset of functionality needed by the snippets is provided.

pd = _optional_import("pandas")  # type: ignore
np = _optional_import("numpy")  # noqa: F841, type: ignore
_optional_import("matplotlib")
_optional_import("matplotlib.pyplot")
_optional_import("geopandas")
_optional_import("PIL")

# If pandas is stubbed in, add a tiny DataFrame placeholder so object creation
# in `_dummy_predictions` works.

if not hasattr(pd, "DataFrame"):
    class _DataFrame(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def apply(self, func, *_, **__):  # mimic pandas apply for our usage
            # Our tests never rely on the return value actually changing
            return self

    pd.DataFrame = _DataFrame  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Helpers to create lightweight dummy outputs that mimic DeepForest results
# -----------------------------------------------------------------------------

def _dummy_predictions():
    """Return a tiny DataFrame that looks like DeepForest output."""
    return pd.DataFrame(
        {
            "xmin": [0],
            "ymin": [0],
            "xmax": [10],
            "ymax": [10],
            "label": [0],
            "score": [0.9],
            "image_path": ["dummy.png"],
        }
    )


# -----------------------------------------------------------------------------
# Create a minimal stub for the deepforest package so docs snippets run instantly
# -----------------------------------------------------------------------------

def _install_deepforest_stub():
    if "deepforest" in sys.modules:
        return  # Already installed

    deepforest_stub = types.ModuleType("deepforest")

    # deepforest.main submodule ------------------------------------------------
    main_mod = types.ModuleType("deepforest.main")

    class _DummyDeepForest:  # pylint: disable=too-few-public-methods
        def load_model(self, *_, **__):
            pass  # no-op

        def predict_image(self, *_, **__):
            return _dummy_predictions()

        def predict_tile(self, *_, **__):
            return _dummy_predictions()

        def predict_batch(self, *_, **__):  # type: ignore
            return _dummy_predictions()

    main_mod.deepforest = _DummyDeepForest  # type: ignore[attr-defined]

    # deepforest.visualize submodule ------------------------------------------
    viz_mod = types.ModuleType("deepforest.visualize")

    def _noop_plot(*_, **__):
        pass

    viz_mod.plot_results = _noop_plot  # type: ignore[attr-defined]

    # get_data helper ----------------------------------------------------------
    def _get_data(path: str):  # pylint: disable=unused-argument
        return "dummy_path.png"

    # Expose in package --------------------------------------------------------
    deepforest_stub.main = main_mod  # type: ignore[attr-defined]
    deepforest_stub.visualize = viz_mod  # type: ignore[attr-defined]
    deepforest_stub.get_data = _get_data  # type: ignore[attr-defined]

    # Register modules in sys.modules
    sys.modules["deepforest"] = deepforest_stub
    sys.modules["deepforest.main"] = main_mod
    sys.modules["deepforest.visualize"] = viz_mod

    # ------------------------------------------------------------------
    # deepforest.datasets.training.BoxDataset stub (used in docs)
    # ------------------------------------------------------------------
    datasets_mod = types.ModuleType("deepforest.datasets")
    training_mod = types.ModuleType("deepforest.datasets.training")

    class _DummyBoxDataset(list):
        def __init__(self, *_, **__):
            super().__init__()

        def collate_fn(self, batch):  # type: ignore
            return batch

    training_mod.BoxDataset = _DummyBoxDataset  # type: ignore[attr-defined]
    datasets_mod.training = training_mod  # type: ignore[attr-defined]
    deepforest_stub.datasets = datasets_mod  # type: ignore[attr-defined]

    sys.modules["deepforest.datasets"] = datasets_mod
    sys.modules["deepforest.datasets.training"] = training_mod


# -----------------------------------------------------------------------------
# Collect python snippets from markdown and execute them
# -----------------------------------------------------------------------------


def _extract_python_blocks(md_text: str):
    pattern = re.compile(r"```python\n(.*?)```", re.S)
    return [m.strip() for m in pattern.findall(md_text)]


DOCS_TO_TEST = list(Path("docs").rglob("*.md"))


# -----------------------------------------------------------------------------
# Pytest entry point
# -----------------------------------------------------------------------------


def test_docs_code_snippets():
    _install_deepforest_stub()

    # Install a minimal torch stub (before docs might import)
    _install_torch_stub()

    for md_path in DOCS_TO_TEST:
        text = md_path.read_text()
        blocks = _extract_python_blocks(text)
        assert blocks, f"No python code blocks found in {md_path}"

        for code in blocks:
            ns: dict = {}
            try:
                exec(code, ns)
            except Exception as exc:  # pylint: disable=broad-except
                raise AssertionError(
                    f"Snippet from {md_path} failed with error: {exc}\nCode:\n{code}"
                ) from exc

def _install_torch_stub():
    """Provide a very small stub of torch and DataLoader used in docs snippets."""
    if "torch" in sys.modules:
        return

    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _DummyLoader(list):  # behaves as iterable with len 0
        def __init__(self, *_, **__):
            super().__init__()

    data_mod.DataLoader = _DummyLoader  # type: ignore[attr-defined]

    utils_mod.data = data_mod  # type: ignore[attr-defined]
    torch_mod.utils = utils_mod  # type: ignore[attr-defined]

    sys.modules["torch"] = torch_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod