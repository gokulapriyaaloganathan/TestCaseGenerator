from __future__ import annotations

from typing import Optional

import gradio as gr

from generate_testcases import (
    AzureDevOpsError,
    AzureOpenAIError,
    ConfigurationError,
    _parse_work_item_ids,
    run_for_user_story,
)


ADO_MODE = "ADO Test Case Generator"
RUN_SUMMARY_SEPARATOR = "=" * 60


def _run_ado_testcase_generation(
    user_story_ids: str,
    plan_id: str,
    parent_suite_id: str,
) -> str:
    user_story_ids = (user_story_ids or "").strip()
    if not user_story_ids:
        return "Please enter one or more User Story IDs."

    plan_value: Optional[int] = None
    raw_plan = (plan_id or "").strip()
    if raw_plan:
        try:
            plan_value = int(raw_plan)
        except ValueError:
            return "Plan ID must be an integer."

    parent_value: Optional[int] = None
    raw_parent = (parent_suite_id or "").strip()
    if raw_parent:
        try:
            parent_value = int(raw_parent)
        except ValueError:
            return "Parent Suite ID must be an integer."

    try:
        ids = _parse_work_item_ids(user_story_ids)
    except ValueError:
        return "Invalid User Story IDs. Example: 12345 or 12345, 12346"

    lines = ["ADO Test Case Generation", RUN_SUMMARY_SEPARATOR]
    failures = 0
    total_created = 0

    for work_item_id in ids:
        lines.append(f"User Story ID: {work_item_id}")
        try:
            result = run_for_user_story(
                work_item_id=work_item_id,
                plan_id=plan_value,
                parent_suite_id=parent_value,
                quiet=True,
            )
        except ConfigurationError as ex:
            failures += 1
            lines.append(f"CONFIG ERROR: {ex}")
            lines.append(RUN_SUMMARY_SEPARATOR)
            continue
        except AzureDevOpsError as ex:
            failures += 1
            lines.append(f"ADO ERROR: {ex}")
            lines.append(RUN_SUMMARY_SEPARATOR)
            continue
        except AzureOpenAIError as ex:
            failures += 1
            lines.append(f"AZURE OPENAI ERROR: {ex}")
            lines.append(RUN_SUMMARY_SEPARATOR)
            continue
        except Exception as ex:
            failures += 1
            lines.append(f"UNEXPECTED ERROR: {type(ex).__name__}: {ex}")
            lines.append(RUN_SUMMARY_SEPARATOR)
            continue

        title = result.get("story_title") or "(unknown)"
        lines.append(f"Title: {title}")

        story_url = result.get("story_url")
        if story_url:
            lines.append(f"Open User Story: {story_url}")

        if result.get("skipped"):
            lines.append(result.get("skip_reason") or "Skipped.")
            lines.append(RUN_SUMMARY_SEPARATOR)
            continue

        created_ids = result.get("created_test_case_ids") or []
        total_created += len(created_ids)

        if created_ids:
            lines.append("Created Test Case IDs: " + ", ".join(map(str, created_ids)))
        else:
            lines.append("No test cases were created.")

        lines.append(RUN_SUMMARY_SEPARATOR)

    lines.append(
        f"Completed. Stories processed: {len(ids)} | Failures: {failures} | Total created test cases: {total_created}"
    )
    return "\n".join(lines)


def _run_selected_generator(
    generator_type: str,
    user_story_ids: str,
    plan_id: str,
    parent_suite_id: str,
) -> str:
    if generator_type != ADO_MODE:
        return "Please select a valid generator type."
    return _run_ado_testcase_generation(user_story_ids, plan_id, parent_suite_id)


def _clear_all_fields() -> tuple[str, str, str, str, str]:
    return (
        ADO_MODE,
        "",
        "",
        "",
        "",
    )


def create_ui() -> gr.Blocks:
    with gr.Blocks(title=" ADO Test Case Generator") as demo:
        gr.Markdown("## ADO Test Case Generator ")

        with gr.Row():
            with gr.Column():
                generator_type = gr.Dropdown(
                    choices=[ADO_MODE],
                    value=ADO_MODE,
                    label="Select Generator",
                )

                user_story_ids = gr.Textbox(
                    label="User Story ID(s)",
                    placeholder="e.g. 12345 or 12345, 12346",
                    lines=1,
                )
                plan_id = gr.Textbox(
                    label="Plan ID",
                    placeholder="",
                    lines=1,
                )
                parent_suite_id = gr.Textbox(
                    label="Parent Suite ID",
                    placeholder="",
                    lines=1,
                )

                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column():
                output = gr.Textbox(label="Run Summary", lines=20, max_lines=50)

        generate_button.click(
            _run_selected_generator,
            inputs=[generator_type, user_story_ids, plan_id, parent_suite_id],
            outputs=[output],
        )

        clear_button.click(
            _clear_all_fields,
            inputs=[],
            outputs=[generator_type, user_story_ids, plan_id, parent_suite_id, output],
        )

    return demo


def main() -> None:
    app = create_ui()
    app.launch(theme=gr.themes.Default())


if __name__ == "__main__":
    main()
