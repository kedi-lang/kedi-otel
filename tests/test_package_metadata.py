from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path

import kedi
from packaging.requirements import Requirement
from packaging.version import Version

from opentelemetry.instrumentation.kedi import KediInstrumentor

ROOT = Path(__file__).resolve().parents[1]


def _kedi_requirement(declarations: Iterable[str]) -> Requirement:
    return next(
        requirement for requirement in map(Requirement, declarations) if requirement.name == "kedi"
    )


def test_instrumented_kedi_range_is_the_package_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = _kedi_requirement(KediInstrumentor().instrumentation_dependencies())

    assert declared.specifier == _kedi_requirement(project["dependencies"]).specifier
    assert Version(kedi.__version__) in declared.specifier
