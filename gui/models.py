"""Pydantic request/response models."""
from typing import Any, Optional
from pydantic import BaseModel


class McpServerIn(BaseModel):
    name: str
    url: str
    transport: str = "streamable_http"
    notes: str = ""
    env_json: str = "{}"


class AgentIn(BaseModel):
    name: str
    file_path: str
    description: str = ""


class PolicyIn(BaseModel):
    name: str
    description: str = ""
    policy_json: str          # raw JSON string of the policy dict


class TestCaseIn(BaseModel):
    name: str
    description: str = ""
    policy_name: str
    agent_name: str
    prompt: str
    expected_pass: bool = True


class SuiteIn(BaseModel):
    name: str
    description: str = ""
    test_case_names: list[str] = []


class RunTestIn(BaseModel):
    test_case_name: str


class RunSuiteIn(BaseModel):
    suite_name: str
