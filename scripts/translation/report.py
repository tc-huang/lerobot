# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).parent
PR_BODY_TEMPLATE_NAME = "pr_body.md.jinja"
DOC_BUILDER_LANGUAGE = {"zh-hans": "zh-CN", "zh-hant": "zh-TW"}
DIFF_BUDGET_BYTES = 30000
PR_BODY_BUDGET_BYTES = 60000
ACTION_EMOJI = {"added": "🆕", "updated": "🔄", "removed": "🗑️", "skipped": "⚠️", "unchanged": "✅"}
ACTION_LABEL = {
    "added": "Added",
    "updated": "Updated",
    "removed": "Removed",
    "skipped": "Needs attention",
    "unchanged": "Unchanged",
}
ERROR_EMOJI = "❌"


def write_text(text_path: Path, text: str) -> None:
    text_path.write_text(text, encoding="utf-8", newline="\n")


def diff_body(diff: str) -> str:
    _, hunk, rest = diff.partition("@@")
    return (hunk + rest).strip() if hunk else diff.strip()


def fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


@dataclass(frozen=True)
class ReviewUrls:
    translation: str
    english: str
    repo: str
    prompt: str
    general_prompt: str
    sync_script: str
    preview_root: str | None
    preview_pending: bool


@dataclass(frozen=True)
class DocView:
    name: str  # the page, a source file name without its suffix
    emoji: str  # what happened to the page, or ❌ when it raised
    title_only: bool  # the English sidebar title changed without a content change
    translation: str  # link to the updated page, "—" once the page is removed
    translation_html: str  # the same link as an anchor for a <summary> line
    english: str  # link to the English page, "—" once the page is removed
    english_html: str  # the same link as an anchor, again for the <summary> line
    translated_from: str  # commit link for the English this translation was made from
    english_now: str  # commit link for the English as this run found it
    changed: bool  # whether there is an English diff stat to show at all
    change_plain: str  # English diff stat without Markdown formatting, for the <summary> line
    english_changes: str  # English commits and line changes since the last translation
    error: str | None  # what the page raised, None when it only failed the structure check
    diff: str | None  # the English diff, None when there is none or it did not fit the body
    fence: str  # a backtick fence long enough to wrap that diff
    command: str | None  # the git diff that reproduces it, None when there is no diff to reproduce
    note: str  # the ", diff not inlined" suffix for the <summary> line, empty otherwise


def _link(label: str, url: str | None) -> str:
    return f"[{label}]({url})" if url else "—"


# Markdown inside a <summary> line is not rendered, so the collapsed rows need real anchors.
def _html_link(label: str, url: str | None) -> str:
    return f'<a href="{url}">{label}</a>' if url else "—"


def _commit(commit: dict | None, repo_url: str, prefix: str = "") -> str:
    if commit is None:
        return "—"
    return f"{prefix}[`{commit['commit'][:7]}`]({repo_url}/commit/{commit['commit']}) {commit['date'][:10]}"


def _doc_view(doc: dict, urls: ReviewUrls, lang_name: str, inlined: bool) -> DocView:
    name = Path(doc["file_name"]).with_suffix("").as_posix()
    action, removed = doc["action"], doc["action"] == "removed"
    source_diff = doc["source_diff"]
    stat = doc["diff_stat"]
    title_only = doc.get("title_only", False)
    if title_only:
        change_plain = "title only"
    elif stat:
        change_plain = f"+{stat[0]} −{stat[1]}"
    else:
        change_plain = "—"
    change = f"`{change_plain}`" if stat or title_only else "—"
    commits = doc["commits_since"]
    url_parts = {"name": name, "file_name": doc["file_name"]}
    translation_url = None if removed else urls.translation.format(**url_parts)
    english_url = None if removed else urls.english.format(**url_parts)
    return DocView(
        name=name,
        emoji=ERROR_EMOJI if doc["error"] else ACTION_EMOJI[action],
        title_only=title_only,
        translation=_link(lang_name, translation_url),
        translation_html=_html_link(lang_name, translation_url),
        english=_link("English", english_url),
        english_html=_html_link("English", english_url),
        translated_from=_commit(doc["translated_from"], urls.repo),
        english_now=_commit(doc["english_now"], urls.repo, "removed in " if removed else ""),
        changed=bool(stat) or title_only,
        change_plain=change_plain,
        english_changes=f"{plural(commits, 'commit')}, {change}" if commits else change,
        error=doc["error"],
        diff=diff_body(source_diff) if source_diff and inlined else None,
        fence=fence(diff_body(source_diff)) if source_diff else "```",
        command=(
            f"git diff {doc['translated_from']['commit'][:7]} {doc['english_now']['commit'][:7]}"
            f" -- docs/source/{doc['file_name']}"
            if source_diff and doc["translated_from"] and doc["english_now"]
            else None
        ),
        note="" if inlined or not source_diff else ", diff not inlined",
    )


def _inlined_names(docs: list[dict], budget: int) -> set[str]:
    sizes = sorted(
        (len(diff_body(doc["source_diff"]).encode("utf-8")), doc["file_name"]) for doc in _diffable(docs)
    )
    names, spent = set(), 0
    for size, file_name in sizes:
        if spent + size > budget:
            break
        names.add(file_name)
        spent += size
    return names


def _diffable(docs: list[dict]) -> list[dict]:
    return [doc for doc in docs if doc["source_diff"] and doc["action"] == "updated"]


def _summary_rows(docs: list[dict]) -> list[dict]:
    rows = []
    for action, label in ACTION_LABEL.items():
        names = [Path(doc["file_name"]).with_suffix("").as_posix() for doc in docs if doc["action"] == action]
        listed = ", ".join(f"`{name}`" for name in names) if names and action != "unchanged" else "—"
        rows.append(
            {
                "emoji": ACTION_EMOJI[action],
                "label": label,
                "count": len(names),
                "names": listed,
            }
        )
    return rows


def _to_review(needs_attention: int, read_in_full: int, changes: int) -> str:
    parts = []
    if needs_attention:
        parts.append(f"{plural(needs_attention, 'page')} need a decision")
    if read_in_full:
        parts.append(f"{plural(read_in_full, 'page')} to read in full")
    if changes:
        parts.append(f"English changes on {plural(changes, 'page')}")
    return ", plus ".join([", ".join(parts[:-1]), parts[-1]]) if len(parts) > 1 else "".join(parts)


def build_context(report: dict, urls: ReviewUrls, inlined: set[str]) -> dict:
    docs = report["docs"]
    lang_name = report["lang_name"]
    views = {
        action: [
            _doc_view(doc, urls, lang_name, doc["file_name"] in inlined)
            for doc in docs
            if doc["action"] == action
        ]
        for action in ACTION_EMOJI
    }
    return {
        "lang_tag": report["lang_tag"],
        "doc_builder_language": DOC_BUILDER_LANGUAGE[report["lang_tag"]],
        "model": report["model"],
        "source": _commit(report["source"], urls.repo),
        "last_sync": report["last_sync"]["date"][:10] if report["last_sync"] else None,
        "last_sync_commit": _commit(report["last_sync"], urls.repo),
        "preview_url": f"{urls.preview_root}/index" if urls.preview_root else None,
        "has_preview": urls.preview_root is not None,
        "preview_pending": urls.preview_pending,
        "prompt_url": urls.prompt,
        "general_prompt_url": urls.general_prompt,
        "sync_script_url": urls.sync_script,
        "rows": _summary_rows(docs),
        "total": len(docs),
        "needs_attention": views["skipped"],
        "pages": views["updated"] + views["added"] + views["removed"],
        "read_in_full": views["added"],
        "changes": views["updated"],
        "counts": {action: len(group) for action, group in views.items()},
        "to_review": _to_review(len(views["skipped"]), len(views["added"]), len(views["updated"])),
    }


def render(report: dict, urls: ReviewUrls) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=False,  # nosec B701: the rendered report is Markdown, HTML escaping would corrupt it
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    pr_body_template = env.get_template(PR_BODY_TEMPLATE_NAME)

    diffs = {doc["file_name"]: diff_body(doc["source_diff"]) for doc in _diffable(report["docs"])}
    inlined = _inlined_names(report["docs"], DIFF_BUDGET_BYTES)
    pr_body = pr_body_template.render(build_context(report, urls, inlined))
    while len(pr_body.encode("utf-8")) > PR_BODY_BUDGET_BYTES and inlined:
        largest = max(inlined, key=lambda name: len(diffs[name].encode("utf-8")))
        inlined.remove(largest)
        pr_body = pr_body_template.render(build_context(report, urls, inlined))
    body_size = len(pr_body.encode("utf-8"))
    if body_size > PR_BODY_BUDGET_BYTES:
        raise ValueError(f"PR body is {body_size} bytes, exceeding the {PR_BODY_BUDGET_BYTES}-byte limit")
    return pr_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preview-url", help="root of the deployment built for this run")
    parser.add_argument("--preview-pending", action="store_true")
    parser.add_argument("--english-url", help="root of the English docs built for this run")
    parser.add_argument("--repo-url", default="https://github.com/huggingface/lerobot")
    parser.add_argument(
        "--source-ref", required=True, help="ref containing the translation scripts and prompts"
    )
    parser.add_argument("--lang-tag", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--pr-body", type=Path, required=True)
    args = parser.parse_args()

    if args.preview_pending and not args.preview_url:
        parser.error("--preview-pending requires --preview-url")
    repo_url = args.repo_url.rstrip("/")
    source_root = f"{repo_url}/blob/{args.source_ref}"
    file_translation_url = f"{repo_url}/blob/{args.head_ref}/docs/source/{args.lang_tag}/{{file_name}}"
    file_english_url = f"{repo_url}/blob/{args.base_ref}/docs/source/{{file_name}}"
    if args.preview_url:
        preview_root = args.preview_url.rstrip("/")
        translation_url = f"{preview_root}/{{name}}"
        english_url = f"{args.english_url.rstrip('/')}/{{name}}" if args.english_url else file_english_url
    else:
        preview_root = None
        translation_url = file_translation_url
        english_url = file_english_url
    urls = ReviewUrls(
        translation=translation_url,
        english=english_url,
        repo=repo_url,
        prompt=f"{source_root}/docs/translation-prompts/{args.lang_tag}.md",
        general_prompt=f"{source_root}/scripts/translation/prompt.md.jinja",
        sync_script=f"{source_root}/scripts/translation/sync.py",
        preview_root=preview_root,
        preview_pending=args.preview_pending,
    )
    write_text(args.pr_body, render(json.loads(args.report.read_text(encoding="utf-8")), urls))


if __name__ == "__main__":
    main()
