import sys
import types
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def monkeypatch_webui(monkeypatch, tmp_path):
    import torch

    fake_webui = {
        "modules": {},
        "modules.scripts": {"Script": object, "basedir": lambda: str(tmp_path)},
        "modules.devices": {"device": torch.device("cpu")},
        "modules.processing": {"fix_seed": Mock()},
        "modules.shared": {
            "opts": types.SimpleNamespace(
                dp_auto_purge_cache=False,
                dp_ignore_whitespace=True,
                dp_limit_jinja_prompts=False,
                dp_magicprompt_batch_size=1,
                dp_parser_variant_end="}",
                dp_parser_variant_start="{",
                dp_parser_wildcard_wrap="__",
                dp_wildcard_manager_no_dedupe=False,
                dp_wildcard_manager_no_sort=False,
                dp_wildcard_manager_shuffle=False,
                dp_write_prompts_to_file=False,
                dp_write_raw_template=False,
            ),
        },
        "modules.script_callbacks": {
            "ImageSaveParams": object,
            "__getattr__": MagicMock(),
        },
        "modules.generation_parameters_copypaste": {
            "parse_generation_parameters": Mock(),
        },
    }

    for module_name, contents in fake_webui.items():
        mod = types.ModuleType(module_name)
        for name, obj in contents.items():
            setattr(mod, name, obj)
        monkeypatch.setitem(sys.modules, module_name, mod)


def test_script(monkeypatch, monkeypatch_webui, processing):
    from scripts.dynamic_prompting import Script

    # monkeypatch.setattr(Script, "__init__", lambda *args, **kwargs: None)
    s = Script()
    processing.set_prompt_for_test("{red|green|blue} ball")
    s.process(
        p=processing,
        is_enabled=True,
        is_combinatorial=True,
        combinatorial_batches=1,
        is_magic_prompt=False,
        is_feeling_lucky=False,
        is_attention_grabber=False,
        min_attention=0,
        max_attention=1,
        magic_prompt_length=0,
        magic_temp_value=1,
        use_fixed_seed=False,
        unlink_seed_from_prompt=False,
        disable_negative_prompt=False,
        enable_jinja_templates=False,
        no_image_generation=False,
        max_generations=None,
        magic_model="magic",
        magic_blocklist_regex=None,
    )
    assert processing.all_prompts == ["red ball", "green ball", "blue ball"]
