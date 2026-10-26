"""Strict declarations and the single JSON result contract (no legacy readers)."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA = "atomiverse.scientific-benchmark/1"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, serialize_by_alias=True)


class System(Contract):
    id: str
    label: str
    atomic_numbers: list[int]
    positions_angstrom: list[list[float]]
    charge: int = 0
    spin: int = 0
    tiers: list[str] = ["smoke", "acceptance", "performance"]

    @model_validator(mode="after")
    def dimensions(self) -> System:
        if not self.atomic_numbers or len(self.atomic_numbers) != len(self.positions_angstrom):
            raise ValueError("positions must match the nonempty species list")
        if any(len(p) != 3 for p in self.positions_angstrom):
            raise ValueError("positions must have three components")
        n = sum(self.atomic_numbers) - self.charge
        if self.spin < 0 or self.spin > n or (n - self.spin) % 2:
            raise ValueError("inconsistent charge/spin/electron count")
        return self


class Participant(Contract):
    id: str
    engine: str
    backend: Literal["cpu", "cuda"]
    role: Literal["atv", "oracle"]
    entrypoint: str
    options: dict[str, Any] = {}
    required: bool = False
    available: bool = True
    unavailable_reason: str | None = None


class Check(Contract):
    observable: str
    unit: str
    atol: float = Field(ge=0)
    rtol: float = Field(default=0, ge=0)
    justification: str


class Row(Contract):
    id: str
    label: str
    capability: str
    task: str
    model: dict[str, Any]
    systems: list[System]
    participants: list[Participant]
    reference: str
    checks: list[Check]
    cuda_only: bool = False

    @model_validator(mode="after")
    def identities(self) -> Row:
        for entries in (self.systems, self.participants):
            ids = [e.id for e in entries]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate identity in method row")
        refs = [p for p in self.participants if p.id == self.reference and p.role == "oracle"]
        if len(refs) != 1:
            raise ValueError("row must name exactly one independent reference")
        if not self.checks or not refs[0].required:
            raise ValueError("checks and a required independent reference are mandatory")
        return self


class Method(Contract):
    id: str
    rows: list[Row]


class Observation(Contract):
    phase: Literal["cold", "warm"]
    repeat: int
    status: str
    reason: str | None = None
    seconds: float | None = Field(default=None, gt=0)
    process_seconds: float | None = Field(default=None, gt=0)
    observables: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    artifact: str | None = None


class Cell(Contract):
    row_id: str
    system_id: str
    participant_id: str
    required: bool
    status: str = "not_run"
    reason: str | None = None
    observations: list[Observation] = []
    numerical: Literal["pass", "fail", "unverified"] = "unverified"
    checks: list[dict[str, Any]] = []
    timing: dict[str, Any] = {}
    speedup: float | None = None
    speedup_reason: str | None = None
    cuda_only: Literal["unverified", "pass", "fail", "not_applicable"] = "not_applicable"


class Bundle(Contract):
    schema_version: Literal["atomiverse.scientific-benchmark/1"] = Field(default=SCHEMA, alias="schema")
    campaign_id: str
    started: str
    finished: str | None = None
    tier: str
    repeats: int = Field(ge=1)
    warmup: int = Field(ge=0)
    threads: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    profile: str
    source: dict[str, Any]
    environment: dict[str, Any]
    rows: list[Row]
    cells: list[Cell]
    protocol: dict[str, Any]
    summary: dict[str, Any] = {}
    roadmap: list[dict[str, Any]] = []

    @model_validator(mode="after")
    def coverage(self) -> Bundle:
        if len({r.id for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate method row IDs")
        expected = {(r.id, s.id, p.id) for r in self.rows for s in r.systems for p in r.participants}
        got = [(c.row_id, c.system_id, c.participant_id) for c in self.cells]
        if len(got) != len(set(got)) or set(got) != expected:
            raise ValueError("result cells must exactly cover the declared matrix")
        policies = {(r.id, s.id, p.id): p.required for r in self.rows for s in r.systems for p in r.participants}
        if any(c.required != policies[c.row_id, c.system_id, c.participant_id] for c in self.cells):
            raise ValueError("cell required policy differs from the declared matrix")
        json.dumps(self.model_dump(), allow_nan=False)
        return self
