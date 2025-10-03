import pytest
from deepforest import main
from deepforest.model import CropModel


ORG = "weecology"


def _list_org_models():
    try:
        from huggingface_hub import HfApi
    except Exception:
        pytest.skip("huggingface_hub not installed; skipping Hub loading test")

    api = HfApi()
    try:
        infos = api.list_models(author=ORG)
    except Exception as exc:
        pytest.skip(f"Could not list models for org {ORG}: {exc}")

    repo_ids = []
    for info in infos:
        repo_id = getattr(info, "id", None) or getattr(info, "modelId", None)
        if not repo_id:
            # Skip unparseable entries
            continue
        repo_ids.append(repo_id)

    if not repo_ids:
        pytest.skip(f"No models found for org {ORG}")

    return repo_ids


@pytest.mark.parametrize("repo_id", _list_org_models())
def test_load_all_weecology_models(repo_id):
    """Load all public models in the weecology org from the HF Hub.
    """
    if "cropmodel" in repo_id.lower():

        model = CropModel.load_model(repo_id=repo_id)
        assert model is not None
        # light sanity: has label_dict and num_classes after load
        assert getattr(model, "label_dict", None) is not None
        assert getattr(model, "num_classes", None) is not None
    else:
        df = main.deepforest()
        df.load_model(model_name=repo_id)
        assert df.model is not None
        # detection models should have label_dict on the underlying model
        assert getattr(df.model, "label_dict", None) is not None
