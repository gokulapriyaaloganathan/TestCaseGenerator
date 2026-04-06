import base64
import json
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

import requests


CONFIG_PATH = Path(__file__).with_name("config.json")
PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompt_template.txt")
ADO_API_VERSION = "7.1-preview.3"
AZURE_OPENAI_API_VERSION = "2024-02-01"
ADO_WIT_RELATION_TYPES_API_VERSION = "7.1-preview.1"


# Global settings applied to all Azure DevOps REST calls made via `requests`.
# These are set when `load_config()` is called.
_REQUESTS_VERIFY: Any = True
_ADO_ALLOW_REDIRECTS: bool = False


HELP_TEXT = "1 . ADO Test Case Generation"


def _set_global_requests_tls_from_config_raw(raw: Dict[str, Any]) -> None:
    global _REQUESTS_VERIFY

    insecure = bool(raw.get("insecure_skip_tls_verify", False))
    ca_bundle = raw.get("ado_ca_bundle_path") or raw.get("azure_openai_ca_bundle_path")

    if ca_bundle:
        _REQUESTS_VERIFY = str(ca_bundle)
    elif insecure:
        _REQUESTS_VERIFY = False
        try:
            import urllib3  # type: ignore

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    else:
        _REQUESTS_VERIFY = True


def _azure_openai_chat_completions_url(*, endpoint: str, deployment: str) -> str:
    endpoint = endpoint.rstrip("/") + "/"
    return (
        f"{endpoint}openai/deployments/{deployment}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )


def _azure_openai_raw_ping(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    ca_bundle_path: Optional[str] = None,
    insecure_skip_tls_verify: bool = False,
) -> str:
    """Attempt a tiny raw REST call to Azure OpenAI for debugging.

    Returns a short human-readable status string (never raises).
    """

    url = _azure_openai_chat_completions_url(endpoint=endpoint, deployment=deployment)
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0,
        "max_tokens": 5,
    }

    verify: Any = True
    if ca_bundle_path:
        verify = ca_bundle_path
    elif insecure_skip_tls_verify:
        verify = False

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20, verify=verify)
        # Don’t dump full body (may include details); keep it short.
        body = (resp.text or "").strip().replace("\r", " ").replace("\n", " ")
        if len(body) > 500:
            body = body[:500] + "…"
        return f"RAW REST ping: HTTP {resp.status_code} | {body}"
    except Exception as ex:
        return f"RAW REST ping failed: {type(ex).__name__}: {ex}"


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the user prompt template from a file.

    The template uses $DESCRIPTION and $ACCEPTANCE_CRITERIA placeholders.
    """

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigurationError(f"Missing prompt template file: {path}") from e
    except OSError as e:
        raise ConfigurationError(f"Unable to read prompt template file: {path} ({e})") from e

    if not content.strip():
        raise ConfigurationError(f"Prompt template file is empty: {path}")

    return content


class ConfigurationError(RuntimeError):
    pass


class AzureDevOpsError(RuntimeError):
    pass


class AzureOpenAIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_deployment: str

    ado_org: str
    ado_project: str
    ado_pat: str

    ado_test_plan_id: Optional[int] = None
    ado_test_suite_id: Optional[int] = None
    # Parent static suite under which to create per-user-story suites.
    ado_parent_static_suite_id: Optional[int] = None
    # Assign configuration(s) when adding to a suite (test points / Execute depend on this).
    # Provide either ids directly OR provide names and let the tool resolve to ids.
    ado_test_configuration_ids: Optional[List[int]] = None
    ado_test_configuration_names: Optional[List[str]] = None

    # Optional TLS CA bundle path for Azure DevOps REST calls.
    # If not set, the script falls back to azure_openai_ca_bundle_path.
    ado_ca_bundle_path: Optional[str] = None

    azure_openai_ca_bundle_path: Optional[str] = None
    insecure_skip_tls_verify: bool = False
    enable_phi_redaction: bool = True

    # Optional extra fields to set when creating Test Case work items.
    # Example: {"Custom.AIAugmentedTestCaseNEW": "Yes"}
    ado_testcase_fields: Optional[Dict[str, Any]] = None

    # Optional: create/find a suite per user story and skip if already populated.
    ado_create_suite_per_user_story: bool = False
    ado_skip_if_suite_has_testcases: bool = True

    # Optional: add a work item link so each created Test Case is related to the User Story.
    ado_link_testcases_to_user_story: bool = True

    # Optional: override the ADO relation referenceName used for Tested By.
    # If not set, the script will auto-discover supported relation types.
    ado_tested_by_relation_type: Optional[str] = None


def _url_path_segment(value: str) -> str:
    """Encode a URL path segment safely.

    Users sometimes provide already-encoded values (e.g. spaces as %20). Others
    provide raw names with spaces/parentheses. To avoid partial/double encoding,
    normalize by unquoting first and then quoting.
    """

    value = str(value).strip()
    return quote(unquote(value), safe="")


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise ConfigurationError(f"Missing config file: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"config.json is not valid JSON: {e}") from e

    required_keys = [
        "azure_openai_endpoint",
        "azure_openai_key",
        "azure_openai_deployment",
        "ado_org",
        "ado_project",
        "ado_pat",
    ]
    missing = [k for k in required_keys if not raw.get(k)]
    if missing:
        raise ConfigurationError(
            "Missing required config value(s): " + ", ".join(missing)
        )

    # Configure TLS verification for subsequent Azure DevOps REST calls.
    _set_global_requests_tls_from_config_raw(raw)

    endpoint = str(raw["azure_openai_endpoint"]).rstrip("/") + "/"

    return Config(
        azure_openai_endpoint=endpoint,
        azure_openai_key=str(raw["azure_openai_key"]).strip(),
        azure_openai_deployment=str(raw["azure_openai_deployment"]).strip(),
        ado_org=str(raw["ado_org"]).strip(),
        ado_project=str(raw["ado_project"]).strip(),
        ado_pat=str(raw["ado_pat"]).strip(),
        ado_test_plan_id=raw.get("ado_test_plan_id"),
        ado_test_suite_id=raw.get("ado_test_suite_id"),
        ado_parent_static_suite_id=raw.get("ado_parent_static_suite_id"),
        ado_test_configuration_ids=raw.get("ado_test_configuration_ids"),
        ado_test_configuration_names=raw.get("ado_test_configuration_names"),
        ado_ca_bundle_path=raw.get("ado_ca_bundle_path"),
        azure_openai_ca_bundle_path=raw.get("azure_openai_ca_bundle_path"),
        insecure_skip_tls_verify=bool(raw.get("insecure_skip_tls_verify", False)),
        enable_phi_redaction=bool(raw.get("enable_phi_redaction", True)),
        ado_testcase_fields=raw.get("ado_testcase_fields"),

        ado_create_suite_per_user_story=bool(
            raw.get("ado_create_suite_per_user_story", False)
        ),
        ado_skip_if_suite_has_testcases=bool(
            raw.get("ado_skip_if_suite_has_testcases", True)
        ),

        ado_link_testcases_to_user_story=bool(
            raw.get("ado_link_testcases_to_user_story", True)
        ),

        ado_tested_by_relation_type=(
            str(raw.get("ado_tested_by_relation_type")).strip()
            if raw.get("ado_tested_by_relation_type")
            else None
        ),
    )


def ado_list_test_configurations(
    *, org: str, project: str, pat: str
) -> List[Dict[str, Any]]:
    """List all test configurations in the project."""
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/configurations"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code != 200:
        raise AzureDevOpsError(
            f"Failed to list test configurations ({resp.status_code}): {resp.text}"
        )

    payload = resp.json()
    if isinstance(payload, dict) and isinstance(payload.get("value"), list):
        return payload["value"]
    if isinstance(payload, list):
        return payload
    return []


def _resolve_configuration_ids_by_name(
    *, org: str, project: str, pat: str, names: List[str]
) -> List[int]:
    configs = ado_list_test_configurations(org=org, project=project, pat=pat)

    by_name: Dict[str, int] = {}
    for c in configs:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        nm = c.get("name")
        if isinstance(cid, int) and isinstance(nm, str) and nm.strip():
            by_name[nm.strip().lower()] = int(cid)

    resolved: List[int] = []
    missing: List[str] = []
    for n in names:
        key = str(n).strip().lower()
        if not key:
            continue
        if key in by_name:
            resolved.append(by_name[key])
        else:
            missing.append(n)

    if missing:
        available = ", ".join(sorted(list(by_name.keys()))[:25])
        raise AzureDevOpsError(
            "Could not resolve configuration name(s): "
            + ", ".join(missing)
            + ". Available (sample): "
            + available
        )

    # de-dupe
    uniq: List[int] = []
    seen: set[int] = set()
    for cid in resolved:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


def ado_set_suite_default_configurations(
    *,
    org: str,
    project: str,
    pat: str,
    plan_id: int,
    suite_id: int,
    configuration_ids: List[int],
) -> None:
    """Set explicit default configurations on a suite (so Execute gets test points)."""

    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/suites/{suite_id}"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "inheritDefaultConfigurations": False,
        "defaultConfigurations": [{"id": int(i)} for i in configuration_ids],
    }

    resp = requests.patch(
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return
    raise AzureDevOpsError(
        f"Failed to set suite configurations ({resp.status_code}): {resp.text}"
    )


def _ado_work_item_api_url(*, org: str, project: str, work_item_id: int) -> str:
    return (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/wit/workitems/{int(work_item_id)}"
    )


def ado_add_work_item_link(
    *,
    org: str,
    project: str,
    pat: str,
    source_work_item_id: int,
    target_work_item_id: int,
    rel: str,
    comment: str,
) -> None:
    """Add a work item relation link from source -> target."""

    url = _ado_work_item_api_url(
        org=org, project=project, work_item_id=int(source_work_item_id)
    )
    params = {"api-version": ADO_API_VERSION}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json-patch+json",
    }

    target_url = _ado_work_item_api_url(
        org=org, project=project, work_item_id=int(target_work_item_id)
    )

    patch_ops = [
        {
            "op": "add",
            "path": "/relations/-",
            "value": {
                "rel": rel,
                "url": target_url,
                "attributes": {"comment": comment},
            },
        }
    ]

    resp = requests.patch(
        url,
        headers=headers,
        params=params,
        json=patch_ops,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code in (200, 201):
        return

    # If the link already exists, don't fail the whole run.
    # ADO error messages vary by process, so keep this tolerant.
    if resp.status_code == 400:
        text = (resp.text or "").lower()
        if "already" in text and ("relation" in text or "link" in text):
            return

    raise AzureDevOpsError(
        f"Failed to add link '{rel}' ({resp.status_code}) from {source_work_item_id} to {target_work_item_id}: {resp.text}"
    )


_RELATION_TYPES_CACHE: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}


def ado_get_work_item_relation_types(
    *, org: str, project: str, pat: str
) -> List[Dict[str, Any]]:
    cache_key = (org, project)
    if cache_key in _RELATION_TYPES_CACHE:
        return _RELATION_TYPES_CACHE[cache_key]

    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/wit/workitemrelationtypes"
    )
    params = {"api-version": ADO_WIT_RELATION_TYPES_API_VERSION}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code != 200:
        raise AzureDevOpsError(
            f"Failed to list work item relation types ({resp.status_code}): {resp.text}"
        )

    payload = resp.json()
    value = payload.get("value") if isinstance(payload, dict) else None
    types: List[Dict[str, Any]] = value if isinstance(value, list) else []
    _RELATION_TYPES_CACHE[cache_key] = types
    return types


def _discover_tested_by_relation_candidates(
    *, org: str, project: str, pat: str
) -> List[str]:
    try:
        types = ado_get_work_item_relation_types(org=org, project=project, pat=pat)
    except Exception:
        return []

    candidates: List[str] = []
    for t in types:
        if not isinstance(t, dict):
            continue
        ref = t.get("referenceName")
        name = t.get("name")
        if not isinstance(ref, str) or not ref.strip():
            continue

        name_s = str(name).strip().lower() if isinstance(name, str) else ""
        ref_s = ref.strip()
        if "tested" in name_s or "tested" in ref_s.lower():
            candidates.append(ref_s)

    # Prefer obvious TestedBy relations first.
    preferred = [c for c in candidates if "testedby" in c.lower()]
    others = [c for c in candidates if c not in preferred]
    return preferred + others


def ado_add_tested_by_link(
    *,
    org: str,
    project: str,
    pat: str,
    user_story_id: int,
    test_case_id: int,
    relation_type_override: Optional[str] = None,
) -> None:
    """Link the User Story to the Test Case so it appears under Tests/Tested By.

    Many Azure DevOps processes expose this relation in the UI as "Tested By".
    """

    candidates: List[str] = []
    if relation_type_override:
        candidates.append(relation_type_override)

    candidates.extend(
        _discover_tested_by_relation_candidates(org=org, project=project, pat=pat)
    )

    # Common guesses across orgs/processes.
    candidates.extend(
        [
            "Microsoft.VSTS.Common.TestedBy",
            "Microsoft.VSTS.Common.TestedBy-Forward",
            "Microsoft.VSTS.Common.TestedBy-Reverse",
        ]
    )

    # De-dupe while preserving order.
    seen: set[str] = set()
    unique_candidates: List[str] = []
    for c in candidates:
        if isinstance(c, str) and c.strip() and c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    last_err: Optional[AzureDevOpsError] = None
    for rel in unique_candidates:
        try:
            ado_add_work_item_link(
                org=org,
                project=project,
                pat=pat,
                source_work_item_id=int(user_story_id),
                target_work_item_id=int(test_case_id),
                rel=rel,
                comment=f"Tested by Test Case {test_case_id}",
            )
            return
        except AzureDevOpsError as e:
            last_err = e
            # If the relation type is unknown, try the next candidate.
            msg = str(e).lower()
            if "unknown relation type" in msg:
                continue
            raise

    if last_err is not None:
        raise last_err


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _summarize_rule_errors(resp_text: str) -> str:
    payload = _try_parse_json(resp_text)
    if not isinstance(payload, dict):
        return ""

    custom = payload.get("customProperties")
    if not isinstance(custom, dict):
        return ""

    errors = custom.get("RuleValidationErrors")
    if not isinstance(errors, list) or not errors:
        return ""

    missing_fields: List[str] = []
    for err in errors:
        if isinstance(err, dict) and isinstance(err.get("fieldReferenceName"), str):
            missing_fields.append(err["fieldReferenceName"])

    missing_fields = [f for f in missing_fields if f]
    if not missing_fields:
        return ""

    unique = sorted(set(missing_fields))
    return (
        " Required/invalid field(s): "
        + ", ".join(unique)
        + ". Add values under 'ado_testcase_fields' in config.json (example: {\"Custom.X\": \"SomeValue\"})."
    )


def _ado_auth_header(pat: str) -> str:
    # Azure DevOps uses Basic auth with username blank and PAT as password.
    token = base64.b64encode(f":{pat}".encode("utf-8")).decode("utf-8")
    return f"Basic {token}"


def ado_work_item_web_url(*, org: str, project: str, work_item_id: int) -> str:
    return (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_workitems/edit/{work_item_id}"
    )


def ado_get_test_plan(*, org: str, project: str, pat: str, plan_id: int) -> Dict[str, Any]:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/plans/{plan_id}"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return resp.json()
    raise AzureDevOpsError(f"Failed to get test plan ({resp.status_code}): {resp.text}")


def ado_get_test_suites_for_plan(
    *, org: str, project: str, pat: str, plan_id: int
) -> List[Dict[str, Any]]:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/suites"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code != 200:
        raise AzureDevOpsError(
            f"Failed to list test suites ({resp.status_code}): {resp.text}"
        )

    payload = resp.json()
    if isinstance(payload, dict) and isinstance(payload.get("value"), list):
        return payload["value"]
    if isinstance(payload, list):
        return payload
    return []


def ado_create_static_test_suite(
    *,
    org: str,
    project: str,
    pat: str,
    plan_id: int,
    parent_suite_id: int,
    name: str,
    default_configuration_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/suites"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    inherit_defaults = not default_configuration_ids
    body: Dict[str, Any] = {
        "suiteType": "staticTestSuite",
        "name": name,
        "parentSuite": {"id": int(parent_suite_id)},
        "inheritDefaultConfigurations": bool(inherit_defaults),
    }
    if default_configuration_ids:
        body["defaultConfigurations"] = [{"id": int(i)} for i in default_configuration_ids]

    resp = requests.post(
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return resp.json()
    raise AzureDevOpsError(
        f"Failed to create test suite '{name}' ({resp.status_code}): {resp.text}"
    )


def ado_create_requirement_test_suite(
    *,
    org: str,
    project: str,
    pat: str,
    plan_id: int,
    parent_suite_id: int,
    requirement_id: int,
    name: str,
    default_configuration_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/suites"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    inherit_defaults = not default_configuration_ids
    body: Dict[str, Any] = {
        "suiteType": "requirementTestSuite",
        "name": name,
        "parentSuite": {"id": int(parent_suite_id)},
        "requirementId": int(requirement_id),
        "inheritDefaultConfigurations": bool(inherit_defaults),
    }
    if default_configuration_ids:
        body["defaultConfigurations"] = [
            {"id": int(i)} for i in default_configuration_ids
        ]

    resp = requests.post(
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return resp.json()
    raise AzureDevOpsError(
        f"Failed to create requirement suite '{name}' ({resp.status_code}): {resp.text}"
    )


def ado_get_suite_testcases(
    *, org: str, project: str, pat: str, plan_id: int, suite_id: int
) -> List[Dict[str, Any]]:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/Suites/{suite_id}/TestCase"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code != 200:
        raise AzureDevOpsError(
            f"Failed to list suite test cases ({resp.status_code}): {resp.text}"
        )

    payload = resp.json()
    if isinstance(payload, dict) and isinstance(payload.get("value"), list):
        return payload["value"]
    if isinstance(payload, list):
        return payload
    return []


def ado_get_test_suite(
    *, org: str, project: str, pat: str, plan_id: int, suite_id: int
) -> Dict[str, Any]:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/suites/{suite_id}"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return resp.json()
    raise AzureDevOpsError(
        f"Failed to get test suite ({resp.status_code}): {resp.text}"
    )


def ado_set_suite_inherit_default_configurations(
    *, org: str, project: str, pat: str, plan_id: int, suite_id: int, inherit: bool
) -> None:
    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/suites/{suite_id}"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    body = {"inheritDefaultConfigurations": bool(inherit)}
    resp = requests.patch(
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return
    raise AzureDevOpsError(
        f"Failed to update suite inheritDefaultConfigurations ({resp.status_code}): {resp.text}"
    )


def ado_get_work_item(
    *,
    org: str,
    project: str,
    pat: str,
    work_item_id: int,
) -> Dict[str, Any]:
    url = f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}/_apis/wit/workitems/{work_item_id}"
    params = {
        "api-version": ADO_API_VERSION,
        "fields": ",".join(
            [
                "System.Title",
                "System.Description",
                "Microsoft.VSTS.Common.AcceptanceCriteria",
                "System.WorkItemType",
                "System.AreaPath",
                "System.IterationPath",
            ]
        ),
    }
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
    }

    resp = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return resp.json()

    # When PAT auth is missing/invalid, ADO often redirects to an interactive sign-in page.
    if resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("Location", "")
        raise AzureDevOpsError(
            "Azure DevOps returned an auth redirect (HTTP "
            + str(resp.status_code)
            + "). This typically means your PAT is invalid/expired or lacks required scopes. "
            + "Check 'ado_pat', 'ado_org', and 'ado_project' in config.json. "
            + (f"Redirected to: {location}" if location else "")
        )

    # Friendly errors for common situations.
    if resp.status_code == 404:
        raise AzureDevOpsError(
            f"Work item {work_item_id} not found (404). Check the ID, org, and project."
        )

    raise AzureDevOpsError(
        f"Azure DevOps API error ({resp.status_code}): {resp.text}"
    )


def ado_create_test_case(
    *,
    org: str,
    project: str,
    pat: str,
    title: str,
    steps: List[str],
    expected_result: Any,
    additional_fields: Optional[Dict[str, Any]] = None,
    area_path: Optional[str] = None,
    iteration_path: Optional[str] = None,
) -> Dict[str, Any]:
    # Create a Test Case work item using JSON Patch.
    # Endpoint uses "$Test Case" (space encoded).
    url = f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}/_apis/wit/workitems/$Test%20Case"
    params = {"api-version": ADO_API_VERSION}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json-patch+json",
    }

    steps_xml = build_test_steps_xml(steps=steps, expected_result=expected_result)

    patch_ops = [
        {"op": "add", "path": "/fields/System.Title", "value": title},
        # The official steps field is XML in Microsoft.VSTS.TCM.Steps.
        {"op": "add", "path": "/fields/Microsoft.VSTS.TCM.Steps", "value": steps_xml},
    ]

    if isinstance(area_path, str) and area_path.strip():
        patch_ops.append(
            {"op": "add", "path": "/fields/System.AreaPath", "value": area_path}
        )
    if isinstance(iteration_path, str) and iteration_path.strip():
        patch_ops.append(
            {
                "op": "add",
                "path": "/fields/System.IterationPath",
                "value": iteration_path,
            }
        )

    if additional_fields:
        for field_ref, value in additional_fields.items():
            if not isinstance(field_ref, str) or not field_ref.strip():
                continue
            # Avoid sending duplicate/conflicting ops for these.
            if field_ref in ("System.AreaPath", "System.IterationPath"):
                continue
            patch_ops.append(
                {"op": "add", "path": f"/fields/{field_ref}", "value": value}
            )

    resp = requests.post(
        url,
        headers=headers,
        params=params,
        json=patch_ops,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code in (200, 201):
        return resp.json()

    rule_hint = ""
    if resp.status_code == 400:
        rule_hint = _summarize_rule_errors(resp.text)

    raise AzureDevOpsError(
        f"Failed to create Test Case ({resp.status_code}): {resp.text}{rule_hint}"
    )


def ado_add_testcases_to_suite(
    *,
    org: str,
    project: str,
    pat: str,
    plan_id: int,
    suite_id: int,
    test_case_ids: List[int],
    configuration_ids: Optional[List[int]] = None,
) -> None:
    """Add existing Test Case work items to a Test Plan Suite.

    Uses the Test Plans REST API:
    POST .../_apis/testplan/Plans/{planId}/Suites/{suiteId}/TestCase?api-version=7.1
    """

    if not test_case_ids:
        return

    url = (
        f"https://dev.azure.com/{_url_path_segment(org)}/{_url_path_segment(project)}"
        f"/_apis/testplan/Plans/{plan_id}/Suites/{suite_id}/TestCase"
    )
    params = {"api-version": "7.1"}
    headers = {
        "Authorization": _ado_auth_header(pat),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    configuration_ids = configuration_ids or []
    point_assignments = [{"configurationId": int(cid)} for cid in configuration_ids]

    body: List[Dict[str, Any]] = []
    for test_case_id in test_case_ids:
        item: Dict[str, Any] = {"workItem": {"id": int(test_case_id)}}
        if point_assignments:
            item["pointAssignments"] = point_assignments
        body.append(item)

    resp = requests.post(
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=30,
        verify=_REQUESTS_VERIFY,
        allow_redirects=_ADO_ALLOW_REDIRECTS,
    )
    if resp.status_code == 200:
        return
    raise AzureDevOpsError(
        f"Failed to add Test Cases to suite ({resp.status_code}): {resp.text}"
    )


def build_test_steps_xml(*, steps: List[str], expected_result: Any) -> str:
    """Build the XML payload for the Test Case steps field.

    Azure DevOps Test Case steps are stored as an XML document. This function
    supports both shapes:
    - expected_result as a single string (overall expected outcome)
    - expected_result as a list[str] aligned to each step
    """

    def _split_action_expected(step_text: str) -> Tuple[str, str]:
        text = (step_text or "").strip()
        if not text:
            return "", ""

        # Backward compatibility for old prompt output:
        # "Action: ... | Expected: ..."
        match = re.match(
            r"^\s*(?:Action\s*:\s*)?(.*?)\s*\|\s*Expected\s*:\s*(.*?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Strip a leading Action: label if present.
        text = re.sub(r"^\s*Action\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip(), ""

    expected_list: Optional[List[str]] = None
    if isinstance(expected_result, list):
        cleaned = [str(v).strip() for v in expected_result if isinstance(v, str)]
        expected_list = cleaned
    overall_expected = expected_result.strip() if isinstance(expected_result, str) else ""

    root = ET.Element("steps")

    # Action steps
    step_id = 1
    any_step_has_expected = False
    for step_text in steps:
        step_el = ET.SubElement(root, "step", {"id": str(step_id), "type": "ActionStep"})

        action_text, parsed_expected = _split_action_expected(step_text)
        expected_text = parsed_expected
        if expected_list is not None:
            idx = step_id - 1
            if idx < len(expected_list):
                expected_text = expected_list[idx]

        action = ET.SubElement(step_el, "parameterizedString", {"isformatted": "true"})
        action.text = action_text

        expected = ET.SubElement(step_el, "parameterizedString", {"isformatted": "true"})
        expected.text = expected_text
        if expected_text:
            any_step_has_expected = True

        ET.SubElement(step_el, "description")
        step_id += 1

    # Keep backward-compatible behavior only when no step-level expected values exist.
    if overall_expected and not any_step_has_expected:
        verify_el = ET.SubElement(root, "step", {"id": str(step_id), "type": "ActionStep"})
        action = ET.SubElement(verify_el, "parameterizedString", {"isformatted": "true"})
        action.text = "Verify the expected result."

        expected = ET.SubElement(verify_el, "parameterizedString", {"isformatted": "true"})
        expected.text = overall_expected

        ET.SubElement(verify_el, "description")

    # Set metadata attributes last/ id used by ADO.
    root.set("id", "0")
    root.set("last", str(step_id if (overall_expected and not any_step_has_expected) else step_id - 1))

    # ADO accepts the raw XML string
    return ET.tostring(root, encoding="unicode")


def _extract_json_array(text: str) -> str:
    """Best-effort extraction if the model returns extra text around JSON."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise AzureOpenAIError("Azure OpenAI response does not contain a JSON array.")
    return text[start : end + 1]


def _is_db_related_story(*, description: str, acceptance_criteria: str) -> bool:
    text = f"{description}\n{acceptance_criteria}".lower()
    hints = [
        "database",
        " db ",
        "sql",
        "table",
        "column",
        "row",
        "record",
        "stored procedure",
        "procedure",
        "query",
        "insert",
        "update",
        "delete",
        "persist",
        "migration",
    ]
    return any(h in text for h in hints)


def _has_sql_query_text(value: str) -> bool:
    return bool(
        re.search(
            r"\b(select|insert|update|delete|merge|exec|with)\b[\s\S]{0,120}\b(from|into|set|where|values)\b",
            value or "",
            flags=re.IGNORECASE,
        )
    )


def _testcases_include_sql_queries(testcases: List[Dict[str, Any]]) -> bool:
    for item in testcases:
        steps = item.get("steps") if isinstance(item, dict) else None
        if isinstance(steps, list):
            for s in steps:
                if isinstance(s, str) and _has_sql_query_text(s):
                    return True

        expected = item.get("expectedResult") if isinstance(item, dict) else None
        if isinstance(expected, str) and _has_sql_query_text(expected):
            return True
        if isinstance(expected, list):
            for e in expected:
                if isinstance(e, str) and _has_sql_query_text(e):
                    return True

    return False


def generate_testcases_with_azure_openai(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    description: str,
    acceptance_criteria: str,
    prompt_template_path: Optional[Path] = None,
    ca_bundle_path: Optional[str] = None,
    insecure_skip_tls_verify: bool = False,
) -> List[Dict[str, Any]]:
    """Call Azure OpenAI to generate test cases.

    Uses a raw REST call (via `requests`) so TLS settings like
    `azure_openai_ca_bundle_path` / `insecure_skip_tls_verify` are honored
    consistently in corporate network environments.
    """

    url = _azure_openai_chat_completions_url(endpoint=endpoint, deployment=deployment)
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    system_prompt = (
        "You are a QA engineer. Create clear, atomic test cases. "
        "Return ONLY valid JSON and nothing else."
    )

    template_text = load_prompt_template(prompt_template_path) if prompt_template_path else load_prompt_template()
    user_prompt = Template(template_text).safe_substitute(
        {
            "DESCRIPTION": description,
            "ACCEPTANCE_CRITERIA": acceptance_criteria,
        }
    )
    user_prompt = textwrap.dedent(user_prompt).strip()

    db_related = _is_db_related_story(
        description=description,
        acceptance_criteria=acceptance_criteria,
    )
    if db_related:
        user_prompt += (
            "\n\nMANDATORY DB RULES:\n"
            "- Include at least one explicit SQL query in every test case.\n"
            "- Use SQL statements directly in step text (SELECT/INSERT/UPDATE/DELETE).\n"
            "- Include SQL validation for both success and negative scenarios where applicable."
        )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    verify: Any = True
    if ca_bundle_path:
        verify = ca_bundle_path
    elif insecure_skip_tls_verify:
        verify = False

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60, verify=verify)
    except Exception as e:
        diagnostic = _azure_openai_raw_ping(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            ca_bundle_path=ca_bundle_path,
            insecure_skip_tls_verify=insecure_skip_tls_verify,
        )

        hint = ""
        ssl_markers = (
            "CERTIFICATE_VERIFY_FAILED",
            "SSLCertVerificationError",
            "certificate verify failed",
        )
        if (
            not ca_bundle_path
            and not insecure_skip_tls_verify
            and any(m in diagnostic for m in ssl_markers)
        ):
            hint = (
                " TLS hint: set 'azure_openai_ca_bundle_path' to your corporate CA bundle "
                "(recommended) OR set 'insecure_skip_tls_verify': true (unsafe) in config.json."
            )

        raise AzureOpenAIError(
            "Azure OpenAI request failed: "
            f"{type(e).__name__}: {e}. "
            f"Endpoint: {endpoint} Deployment: {deployment}. {diagnostic}{hint}"
        ) from e

    if resp.status_code != 200:
        raise AzureOpenAIError(
            f"Azure OpenAI request failed (HTTP {resp.status_code}): {resp.text}"
        )

    try:
        response_json = resp.json()
        content = str(response_json["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        raise AzureOpenAIError(
            f"Azure OpenAI response missing expected fields: {type(e).__name__}: {e}"
        ) from e

    if not content:
        raise AzureOpenAIError("Azure OpenAI returned an empty response.")

    # Parse JSON strictly; if that fails, attempt to extract the JSON array.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            data = json.loads(_extract_json_array(content))
        except json.JSONDecodeError as e:
            raise AzureOpenAIError(
                f"Azure OpenAI response is not valid JSON: {e}"
            ) from e

    if not isinstance(data, list):
        raise AzureOpenAIError("Azure OpenAI JSON must be a list of test cases.")

    # Validate minimal schema.
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise AzureOpenAIError(f"Test case at index {i} is not an object.")
        if not isinstance(item.get("title"), str) or not item["title"].strip():
            raise AzureOpenAIError(f"Test case at index {i} is missing a 'title'.")
        if not isinstance(item.get("steps"), list) or not item["steps"]:
            raise AzureOpenAIError(f"Test case at index {i} is missing 'steps'.")
        if not all(isinstance(s, str) and s.strip() for s in item["steps"]):
            raise AzureOpenAIError(f"Test case at index {i} has invalid step strings.")

        # Normalize step strings.
        item["steps"] = [str(s).strip() for s in item["steps"]]

        expected_value = item.get("expectedResult")
        if isinstance(expected_value, str):
            item["expectedResult"] = expected_value.strip()
        elif isinstance(expected_value, list):
            # Be tolerant to model output drift: align expected results to step count
            # by trimming extras and padding missing values with empty strings.
            normalized_expected: List[str] = []
            for idx, _ in enumerate(item["steps"]):
                if idx < len(expected_value):
                    value = expected_value[idx]
                    if isinstance(value, str):
                        normalized_expected.append(value.strip())
                    elif value is None:
                        normalized_expected.append("")
                    else:
                        normalized_expected.append(str(value).strip())
                else:
                    normalized_expected.append("")
            item["expectedResult"] = normalized_expected
        else:
            raise AzureOpenAIError(
                f"Test case at index {i} is missing 'expectedResult'."
            )

    if db_related and not _testcases_include_sql_queries(data):
        raise AzureOpenAIError(
            "Generated test cases are missing SQL queries for a DB-related story. "
            "Please regenerate; SQL query steps are mandatory for DB stories."
        )

    return data


def _safe_field(fields: Dict[str, Any], key: str) -> str:
    value = fields.get(key)
    if value is None:
        return ""
    return str(value)


def resolve_test_configuration_ids(config: Config) -> List[int]:
    """Resolve test configuration IDs based on config.

    This is used to ensure test points exist for Execute.
    Never raises; on ADO errors returns an empty list.
    """

    configuration_ids: List[int] = []
    try:
        if config.ado_test_configuration_ids:
            configuration_ids.extend([int(i) for i in config.ado_test_configuration_ids])

        names = [str(n) for n in (config.ado_test_configuration_names or []) if str(n).strip()]
        if not names and not configuration_ids:
            names = ["Windows 10"]

        if names:
            configuration_ids.extend(
                _resolve_configuration_ids_by_name(
                    org=config.ado_org,
                    project=config.ado_project,
                    pat=config.ado_pat,
                    names=names,
                )
            )
    except AzureDevOpsError:
        return []

    # de-dupe preserving order
    return list(dict.fromkeys(configuration_ids))


def _parse_work_item_ids(raw: str) -> List[int]:
    """Parse one or many work item IDs from comma/space/newline separated input."""

    tokens = [t.strip() for t in raw.replace("\n", " ").replace(",", " ").split(" ")]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError("No IDs were provided")

    ids: List[int] = []
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError("IDs must be positive integers")
        ids.append(value)

    # de-dupe preserving order
    return list(dict.fromkeys(ids))


def run_for_user_story(
    *,
    work_item_id: int,
    plan_id: Optional[int] = None,
    parent_suite_id: Optional[int] = None,
    config_path: Optional[Path] = None,
    prompt_template_path: Optional[Path] = None,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run generation + creation for a single ADO User Story ID.

    Designed for non-interactive callers (e.g. a Teams bot). It uses config.json by default.

    Returns a dict with keys:
      - story_url, story_title
      - created_test_case_ids (list[int])
      - plan_id, suite_id
      - skipped (bool), skip_reason (str|None)
    """

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    if work_item_id <= 0:
        raise ValueError("work_item_id must be a positive integer")

    config = load_config(config_path) if config_path else load_config()
    configuration_ids = resolve_test_configuration_ids(config)

    wi = ado_get_work_item(
        org=config.ado_org,
        project=config.ado_project,
        pat=config.ado_pat,
        work_item_id=work_item_id,
    )

    fields = wi.get("fields") or {}
    title = _safe_field(fields, "System.Title")
    description = _safe_field(fields, "System.Description")
    acceptance = _safe_field(fields, "Microsoft.VSTS.Common.AcceptanceCriteria")
    work_item_type = _safe_field(fields, "System.WorkItemType")
    area_path = _safe_field(fields, "System.AreaPath")
    iteration_path = _safe_field(fields, "System.IterationPath")

    if work_item_type and work_item_type.lower() != "user story":
        _log(f"WARNING: Work item type is '{work_item_type}', not 'User Story'. Continuing...")

    if not description and not acceptance:
        raise ConfigurationError("User Story has no Description and no Acceptance Criteria.")

    story_url = ado_work_item_web_url(
        org=config.ado_org, project=config.ado_project, work_item_id=work_item_id
    )

    target_plan_id: Optional[int] = int(plan_id) if plan_id is not None else (
        int(config.ado_test_plan_id) if config.ado_test_plan_id else None
    )
    target_suite_id: Optional[int] = None

    # Suite selection/creation (non-interactive)
    if config.ado_create_suite_per_user_story and target_plan_id:
        # Determine parent suite id.
        effective_parent_suite_id: Optional[int] = None
        if parent_suite_id is not None:
            effective_parent_suite_id = int(parent_suite_id)
        elif config.ado_parent_static_suite_id:
            effective_parent_suite_id = int(config.ado_parent_static_suite_id)
        elif config.ado_test_suite_id:
            # Backward-compatible behavior.
            effective_parent_suite_id = int(config.ado_test_suite_id)

        if not effective_parent_suite_id:
            raise ConfigurationError(
                "Missing parent static suite id. Set ado_parent_static_suite_id in config.json (or pass parent_suite_id)."
            )

        suite_name = f"{work_item_id}"
        suites = ado_get_test_suites_for_plan(
            org=config.ado_org,
            project=config.ado_project,
            pat=config.ado_pat,
            plan_id=target_plan_id,
        )

        for s in suites:
            if not isinstance(s, dict):
                continue

            name = str(s.get("name", "")).strip()
            if name.lower() != suite_name.lower():
                continue

            parent = s.get("parentSuite") or {}
            parent_id = parent.get("id") if isinstance(parent, dict) else None
            if parent_id is not None and int(parent_id) != int(effective_parent_suite_id):
                continue

            req_id = s.get("requirementId")
            if req_id is not None:
                try:
                    if int(req_id) != int(work_item_id):
                        continue
                except Exception:
                    pass

            sid = s.get("id")
            if isinstance(sid, int):
                target_suite_id = sid
                break

        if target_suite_id is None:
            created_suite = ado_create_requirement_test_suite(
                org=config.ado_org,
                project=config.ado_project,
                pat=config.ado_pat,
                plan_id=target_plan_id,
                parent_suite_id=effective_parent_suite_id,
                requirement_id=work_item_id,
                name=suite_name,
                default_configuration_ids=configuration_ids or None,
            )
            if isinstance(created_suite.get("id"), int):
                target_suite_id = int(created_suite["id"])

    if target_suite_id is None and config.ado_test_suite_id:
        target_suite_id = int(config.ado_test_suite_id)

    # Ensure suite has configurations for Execute.
    if target_plan_id and target_suite_id:
        try:
            suite = ado_get_test_suite(
                org=config.ado_org,
                project=config.ado_project,
                pat=config.ado_pat,
                plan_id=int(target_plan_id),
                suite_id=int(target_suite_id),
            )
            current = suite.get("inheritDefaultConfigurations")
            if configuration_ids:
                ado_set_suite_default_configurations(
                    org=config.ado_org,
                    project=config.ado_project,
                    pat=config.ado_pat,
                    plan_id=int(target_plan_id),
                    suite_id=int(target_suite_id),
                    configuration_ids=configuration_ids,
                )
            else:
                if current is False:
                    ado_set_suite_inherit_default_configurations(
                        org=config.ado_org,
                        project=config.ado_project,
                        pat=config.ado_pat,
                        plan_id=int(target_plan_id),
                        suite_id=int(target_suite_id),
                        inherit=True,
                    )
        except AzureDevOpsError:
            pass

    # Idempotency check
    if target_plan_id and target_suite_id and config.ado_skip_if_suite_has_testcases:
        existing = ado_get_suite_testcases(
            org=config.ado_org,
            project=config.ado_project,
            pat=config.ado_pat,
            plan_id=int(target_plan_id),
            suite_id=int(target_suite_id),
        )
        if len(existing) > 0:
            return {
                "story_url": story_url,
                "story_title": title,
                "created_test_case_ids": [],
                "plan_id": target_plan_id,
                "suite_id": target_suite_id,
                "skipped": True,
                "skip_reason": f"Suite already contains {len(existing)} test case(s).",
            }

    _log("Generating test cases with Azure OpenAI...")
    testcases = generate_testcases_with_azure_openai(
        endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_key,
        deployment=config.azure_openai_deployment,
        description=description,
        acceptance_criteria=acceptance,
        prompt_template_path=prompt_template_path,
        ca_bundle_path=config.azure_openai_ca_bundle_path,
        insecure_skip_tls_verify=config.insecure_skip_tls_verify,
    )

    created_ids: List[int] = []
    for tc in testcases:
        created = ado_create_test_case(
            org=config.ado_org,
            project=config.ado_project,
            pat=config.ado_pat,
            title=tc["title"],
            steps=tc["steps"],
            expected_result=tc.get("expectedResult", ""),
            additional_fields=config.ado_testcase_fields,
            area_path=area_path,
            iteration_path=iteration_path,
        )
        created_id = created.get("id")
        if isinstance(created_id, int):
            created_ids.append(created_id)
            if config.ado_link_testcases_to_user_story:
                try:
                    ado_add_tested_by_link(
                        org=config.ado_org,
                        project=config.ado_project,
                        pat=config.ado_pat,
                        user_story_id=work_item_id,
                        test_case_id=created_id,
                        relation_type_override=config.ado_tested_by_relation_type,
                    )
                except AzureDevOpsError:
                    pass

    if created_ids and target_plan_id and target_suite_id:
        ado_add_testcases_to_suite(
            org=config.ado_org,
            project=config.ado_project,
            pat=config.ado_pat,
            plan_id=int(target_plan_id),
            suite_id=int(target_suite_id),
            test_case_ids=created_ids,
            configuration_ids=configuration_ids or None,
        )

    return {
        "story_url": story_url,
        "story_title": title,
        "created_test_case_ids": created_ids,
        "plan_id": target_plan_id,
        "suite_id": target_suite_id,
        "skipped": False,
        "skip_reason": None,
    }


def _build_help_response() -> str:
    return "Help\n\n" + HELP_TEXT


def _chat_response(message: str) -> str:
    text = (message or "").strip()
    lowered = text.lower()
    if lowered in {"help", "/help"}:
        return _build_help_response()

    return (
        "Please type 'help' or click the Help button.\n\n"
        + _build_help_response()
    )


def launch_gradio_ui() -> int:
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: Gradio is not installed. Install it with: pip install gradio")
        return 2

    with gr.Blocks(title="ADO Test Case Assistant") as demo:
        gr.Markdown("## ADO Test Case Assistant")
        chatbot = gr.Chatbot(label="Chat", type="messages")

        with gr.Row():
            message = gr.Textbox(
                label="Message",
                placeholder="Type 'help' or click Help",
                lines=1,
            )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            help_btn = gr.Button("Help")
            clear_btn = gr.Button("Clear")

        def on_send(user_message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
            response = _chat_response(user_message)
            history = history or []
            history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response},
            ]
            return "", history

        def on_help(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
            history = history or []
            return history + [{"role": "assistant", "content": _build_help_response()}]

        send_btn.click(on_send, inputs=[message, chatbot], outputs=[message, chatbot])
        message.submit(on_send, inputs=[message, chatbot], outputs=[message, chatbot])
        help_btn.click(on_help, inputs=[chatbot], outputs=[chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])

    demo.launch()
    return 0


def main() -> int:
    if "--ui" in sys.argv:
        return launch_gradio_ui()

    try:
        config = load_config()
    except ConfigurationError as e:
        print(f"CONFIG ERROR: {e}")
        return 2

    raw_ids = input("Enter Azure DevOps User Story ID(s) (comma or space separated): ").strip()
    try:
        work_item_ids = _parse_work_item_ids(raw_ids)
    except ValueError:
        print("ERROR: Invalid user story id input. Use positive integer IDs (e.g. 12345 or 12345, 12346).")
        return 2

    print("Processing User Story IDs: " + ", ".join(map(str, work_item_ids)))

    # Determine parent suite id interactively if suite-per-story is enabled.
    parent_suite_id: Optional[int] = None
    if config.ado_create_suite_per_user_story and config.ado_test_plan_id:
        default_parent_suite_id: Optional[int] = None
        if config.ado_parent_static_suite_id:
            default_parent_suite_id = int(config.ado_parent_static_suite_id)
        elif config.ado_test_suite_id:
            default_parent_suite_id = int(config.ado_test_suite_id)
            print(
                "NOTE: Using ado_test_suite_id as the default parent static suite id for per-user-story suite creation. "
                "Consider setting ado_parent_static_suite_id in config.json to make this explicit."
            )

        prompt_suffix = f" [{default_parent_suite_id}]" if default_parent_suite_id else ""
        raw_parent = input(
            "Parent suite id (under which to create a requirement suite for this User Story)"
            + prompt_suffix
            + ": "
        ).strip()

        if raw_parent:
            try:
                parent_suite_id = int(raw_parent)
            except ValueError:
                print("ERROR: Parent static suite id must be an integer.")
                return 2
        else:
            parent_suite_id = default_parent_suite_id

    all_created_ids: List[int] = []
    failures = 0

    for work_item_id in work_item_ids:
        print("\n" + "=" * 70)
        print(f"User Story {work_item_id}")

        try:
            result = run_for_user_story(
                work_item_id=work_item_id,
                parent_suite_id=parent_suite_id,
                quiet=False,
            )
        except ConfigurationError as e:
            print(f"ERROR [{work_item_id}]: {e}")
            failures += 1
            continue
        except AzureDevOpsError as e:
            print(f"ADO ERROR [{work_item_id}]: {e}")
            failures += 1
            continue
        except AzureOpenAIError as e:
            print(f"AZURE OPENAI ERROR [{work_item_id}]: {e}")
            failures += 1
            continue

        print(f"Title: {result.get('story_title') or '(unknown)'}")
        if result.get("story_url"):
            print(f"Open User Story: {result['story_url']}")

        if result.get("skipped"):
            print(result.get("skip_reason") or "Skipped.")
            continue

        created_ids = result.get("created_test_case_ids") or []
        all_created_ids.extend([int(i) for i in created_ids])

        if created_ids:
            print("Created Test Case IDs: " + ", ".join(map(str, created_ids)))
        else:
            print("No test cases were created.")

    print("\n" + "=" * 70)
    print(
        "Completed. "
        f"Stories processed: {len(work_item_ids)} | "
        f"Failures: {failures} | "
        f"Total created test cases: {len(all_created_ids)}"
    )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
