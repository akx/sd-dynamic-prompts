from __future__ import annotations

import dataclasses

import pytest


@dataclasses.dataclass
class MockProcessing:
    seed: int
    subseed: int
    all_seeds: list[int]
    all_subseeds: list[int]
    subseed_strength: float
    prompt: str
    negative_prompt: str
    n_iter: int = 1
    batch_size: int = 1
    all_prompts: list[str] = dataclasses.field(default_factory=list)
    all_negative_prompts: list[str] = dataclasses.field(default_factory=list)

    def set_prompt_for_test(self, prompt):
        self.prompt = prompt
        self.all_prompts = [prompt] * self.n_iter * self.batch_size

    def set_negative_prompt_for_test(self, negative_prompt):
        self.negative_prompt = negative_prompt
        self.all_negative_prompts = [negative_prompt] * self.n_iter * self.batch_size


@pytest.fixture
def processing() -> MockProcessing:
    return MockProcessing(
        seed=1000,
        subseed=2000,
        all_seeds=list(range(3000, 3000 + 10)),
        all_subseeds=list(range(4000, 4000 + 10)),
        subseed_strength=0,
        prompt="beautiful sheep",
        negative_prompt="ugly",
        all_prompts=["beautiful"],
        all_negative_prompts=["ugly"],
    )
