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
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from huggingface_hub import InferenceClient
from jinja2 import Environment, FileSystemLoader, StrictUndefined

COMMIT_FORMAT = "%H%x09%cI"
TOCTREE_FILE_NAME = "_toctree.yml"
RECORD_FILE_NAME = "translation-record.json"
LANG_TAG2NAME = {"zh-hant": "Traditional Chinese"}
TRANSLATION_PROMPT_TEMPLATE_NAME = "prompt.md.jinja"
TRANSLATION_PROMPT_TEMPLATE_DIR = Path(__file__).parent
LINE_COUNT_TOLERANCE = 10
BLANK_LINE_COUNT_TOLERANCE = 0
API_KEY_ENV = "TRANSLATION_API_KEY"


def read_text(text_path: Path) -> str:
    return text_path.read_text(encoding="utf-8")


def write_text(text_path: Path, text: str) -> None:
    text_path.write_text(text, encoding="utf-8", newline="\n")


def doc_paths(doc_dir: Path) -> list[Path]:
    return list(doc_dir.glob("*.mdx"))


def doc_file_names(doc_dir: Path) -> set[str]:
    return {path.name for path in doc_paths(doc_dir)}


def page_locals(doc_dir: Path) -> set[str]:
    # How a toctree addresses a page: its path under doc_dir, without the suffix. doc-builder
    # builds .md pages as well and requires each one to be listed there, READMEs excepted.
    return {
        path.relative_to(doc_dir).with_suffix("").as_posix()
        for path in doc_dir.rglob("*")
        if path.suffix in (".md", ".mdx") and not path.stem.endswith("README")
    }


@dataclass(frozen=True)
class Commit:
    commit: str
    date: str


class GitRepo:
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self._root_dir))

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(  # nosec B607: git comes from PATH, the argument list is fixed and never shell-interpreted
            ["git", *args], cwd=self._root_dir, capture_output=True, text=True, check=False
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
        return result

    def _parse_commit(self, line: str) -> Commit | None:
        if not line:
            return None
        commit, date = line.split("\t")
        return Commit(commit=commit, date=date)

    def blob_hash(self, path: Path) -> str:
        return self._run("hash-object", "--", self._relative(path)).stdout.strip()

    def blob_at(self, commit: str, path: Path) -> str | None:
        result = self._run("rev-parse", f"{commit}:{self._relative(path)}", check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def diff_blobs(self, old_blob: str, new_blob: str) -> str:
        return self._run("diff", old_blob, new_blob).stdout

    def diff_stat(self, old_blob: str, new_blob: str) -> tuple[int, int]:
        numstat = self._run("diff", "--numstat", old_blob, new_blob).stdout.split()
        return (int(numstat[0]), int(numstat[1])) if numstat else (0, 0)

    def last_touching_commit(
        self, path: Path, ref: str = "HEAD", exclude: Path | None = None
    ) -> Commit | None:
        pathspec = [self._relative(path)]
        if exclude is not None:
            pathspec.append(f":(exclude){self._relative(exclude)}")
        log = self._run("log", "-1", f"--format={COMMIT_FORMAT}", ref, "--", *pathspec).stdout
        return self._parse_commit(log.strip())

    def count_commits_since(self, old_commit: str, path: Path) -> int:
        revisions = f"{old_commit}..HEAD"
        return int(self._run("rev-list", "--count", revisions, "--", self._relative(path)).stdout.strip())


type Section = tuple[str, list[tuple[str, str]]]


class Toctree:
    def __init__(self, doc_dir: Path) -> None:
        self._toctree_path = doc_dir / TOCTREE_FILE_NAME
        self._toctree = self._load()

    def _load(self) -> list[dict]:
        if not self._toctree_path.exists():
            return []
        return yaml.safe_load(read_text(self._toctree_path)) or []

    def _find(self, doc_file_name: str) -> tuple[dict | None, dict | None]:
        local = Path(doc_file_name).stem
        for section in self._toctree:
            for entry in section["sections"]:
                if entry["local"] == local:
                    return section, entry
        return None, None

    def get_section_and_doc_title(self, doc_file_name: str) -> tuple[str, str] | None:
        section, entry = self._find(doc_file_name)
        if section is None or entry is None:
            return None
        return section["title"], entry["title"]

    @property
    def layout(self) -> list[Section]:
        return [
            (section["title"], [(entry["local"], entry["title"]) for entry in section["sections"]])
            for section in self._toctree
        ]

    @layout.setter
    def layout(self, sections: list[Section]) -> None:
        self._toctree = [
            {
                "sections": [{"local": local, "title": title} for local, title in entries],
                "title": section_title,
            }
            for section_title, entries in sections
            if entries
        ]

    def save(self) -> None:
        yaml_sections = yaml.safe_dump(self._toctree, allow_unicode=True, sort_keys=True, width=4096)
        write_text(self._toctree_path, yaml_sections)


class Docs:
    def __init__(self, doc_dir: Path) -> None:
        self._doc_dir = doc_dir
        self._toctree = Toctree(doc_dir)

    def get_section_and_doc_title(self, doc_file_name: str) -> tuple[str, str] | None:
        return self._toctree.get_section_and_doc_title(doc_file_name)

    def get_doc_content(self, doc_file_name: str) -> str:
        doc_content = read_text(self._doc_dir / doc_file_name)
        return doc_content

    @property
    def file_names(self) -> set[str]:
        file_names = doc_file_names(self._doc_dir)
        return file_names

    @property
    def layout(self) -> list[Section]:
        return self._toctree.layout


class SourceDocs(Docs):
    def __init__(self, doc_en_dir: Path, git: GitRepo) -> None:
        super().__init__(doc_en_dir)
        self._git = git

    def _matching_source_commit(
        self, target_path: Path, source_path: Path, exclude: Path | None = None
    ) -> Commit | None:
        """The source commit that the state of target_path was produced from."""
        target_commit = self._git.last_touching_commit(target_path)
        if target_commit is None:
            return None
        return self._git.last_touching_commit(source_path, ref=target_commit.commit, exclude=exclude)

    def get_current_version(self, doc_file_name: str) -> str:
        version = self._git.blob_hash(self._doc_dir / doc_file_name)
        return version

    def get_current_commit(self, doc_file_name: str) -> Commit | None:
        return self._git.last_touching_commit(self._doc_dir / doc_file_name)

    def get_latest_commit(self, doc_lang_dir: Path) -> Commit | None:
        return self._git.last_touching_commit(self._doc_dir, exclude=doc_lang_dir)

    def get_diff(self, doc_file_name: str, old_version: str) -> str:
        current_version = self.get_current_version(doc_file_name)
        diff = self._git.diff_blobs(old_version, current_version)
        return diff

    def get_diff_stat(self, doc_file_name: str, old_version: str) -> tuple[int, int]:
        return self._git.diff_stat(old_version, self.get_current_version(doc_file_name))

    def get_translated_version(self, doc_lang_dir: Path, doc_file_name: str) -> str | None:
        commit = self._git.last_touching_commit(doc_lang_dir / doc_file_name)
        if commit is None:
            return None
        return self._git.blob_at(commit.commit, self._doc_dir / doc_file_name)

    def get_translated_commit(self, doc_lang_dir: Path, doc_file_name: str) -> Commit | None:
        return self._matching_source_commit(doc_lang_dir / doc_file_name, self._doc_dir / doc_file_name)

    def get_last_sync_commit(self, doc_lang_dir: Path) -> Commit | None:
        return self._matching_source_commit(doc_lang_dir, self._doc_dir, exclude=doc_lang_dir)

    def count_commits_since(self, doc_file_name: str, old_commit: Commit | None) -> int:
        if old_commit is None:
            return 0
        return self._git.count_commits_since(old_commit.commit, self._doc_dir / doc_file_name)


class TranslationRecord:
    def __init__(self, doc_lang_dir: Path, source_docs: "SourceDocs", doc_lang_toctree: Toctree) -> None:
        self._doc_lang_dir = doc_lang_dir
        self._record_path = doc_lang_dir / RECORD_FILE_NAME
        self._source_docs = source_docs

        record = (
            json.loads(read_text(self._record_path))
            if self._record_path.exists()
            else self._build(doc_lang_toctree)
        )
        self._docs = record["docs"]
        self._section_titles = record["section_titles"]

    def _build(self, doc_lang_toctree: Toctree) -> dict:
        docs = {}
        section_titles = {}
        for file_name in sorted(self._source_docs.file_names & doc_file_names(self._doc_lang_dir)):
            doc_lang_titles = doc_lang_toctree.get_section_and_doc_title(file_name)
            docs[file_name] = {
                "source_version": self._source_docs.get_translated_version(self._doc_lang_dir, file_name),
                "title": doc_lang_titles[1] if doc_lang_titles else None,
            }
            doc_en_titles = self._source_docs.get_section_and_doc_title(file_name)
            if doc_en_titles is not None and doc_lang_titles is not None:
                section_titles[doc_en_titles[0]] = doc_lang_titles[0]
        return {"docs": docs, "section_titles": section_titles}

    def get_last_translated_version(self, doc_file_name: str) -> str | None:
        return self._docs.get(doc_file_name, {}).get("source_version")

    def get_doc_title(self, doc_file_name: str) -> str | None:
        return self._docs.get(doc_file_name, {}).get("title")

    def get_section_title(self, doc_en_section_title: str) -> str | None:
        return self._section_titles.get(doc_en_section_title)

    def set_section_title(self, doc_en_section_title: str, doc_section_title: str) -> None:
        self._section_titles[doc_en_section_title] = doc_section_title

    def add(self, doc_file_name: str, doc_title: str) -> None:
        self._docs[doc_file_name] = {
            "source_version": self._source_docs.get_current_version(doc_file_name),
            "title": doc_title,
        }

    def update(self, doc_file_name: str) -> None:
        self._docs[doc_file_name]["source_version"] = self._source_docs.get_current_version(doc_file_name)

    def remove(self, doc_file_name: str) -> None:
        self._docs.pop(doc_file_name, None)

    def save(self) -> None:
        record = {"docs": self._docs, "section_titles": self._section_titles}
        write_text(self._record_path, json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True))


class TargetDocs(Docs):
    def __init__(self, doc_lang_dir: Path, source_docs: SourceDocs) -> None:
        super().__init__(doc_lang_dir)
        self._source_docs = source_docs
        self._record = TranslationRecord(doc_lang_dir, source_docs, self._toctree)

    def _reordered_layout(self) -> list[Section]:
        pages = page_locals(self._doc_dir)
        # contributing.md is a page this script never translates, so it keeps its existing title
        current = {local: title for _, entries in self._toctree.layout for local, title in entries}
        return [
            (
                self._record.get_section_title(section_title) or section_title,
                [
                    (local, self._record.get_doc_title(f"{local}.mdx") or current.get(local, title))
                    for local, title in entries
                    if local in pages
                ],
            )
            for section_title, entries in self._source_docs.layout
        ]

    def _save(self) -> None:
        self._toctree.layout = self._reordered_layout()
        self._toctree.save()
        self._record.save()

    def _write(self, doc_file_name: str, doc_content: str) -> None:
        write_text(self._doc_dir / doc_file_name, doc_content)

    def _remove(self, doc_file_name: str) -> None:
        doc_path = self._doc_dir / doc_file_name
        doc_path.unlink(missing_ok=True)

    def _need_update(self, doc_file_name: str) -> bool:
        return self._record.get_last_translated_version(
            doc_file_name
        ) != self._source_docs.get_current_version(doc_file_name)

    def get_last_translated_version(self, doc_file_name: str) -> str | None:
        return self._record.get_last_translated_version(doc_file_name)

    def get_translated_commit(self, doc_file_name: str) -> Commit | None:
        return self._source_docs.get_translated_commit(self._doc_dir, doc_file_name)

    def get_last_sync_commit(self) -> Commit | None:
        return self._source_docs.get_last_sync_commit(self._doc_dir)

    def get_latest_source_commit(self) -> Commit | None:
        return self._source_docs.get_latest_commit(self._doc_dir)

    def get_section_title(self, doc_en_section_title: str) -> str | None:
        return self._record.get_section_title(doc_en_section_title)

    def set_section_title(self, doc_en_section_title: str, doc_section_title: str) -> None:
        self._record.set_section_title(doc_en_section_title, doc_section_title)

    def add_doc(self, doc_file_name: str, doc_content: str, doc_title: str) -> None:
        self._record.add(doc_file_name, doc_title)
        self._write(doc_file_name, doc_content)
        self._save()

    def remove_doc(self, doc_file_name: str) -> None:
        self._record.remove(doc_file_name)
        self._remove(doc_file_name)
        self._save()

    def update_doc(self, doc_file_name: str, doc_content: str) -> None:
        self._record.update(doc_file_name)
        self._write(doc_file_name, doc_content)
        self._save()

    @property
    def files_to_add(self):
        return sorted(self._source_docs.file_names - self.file_names)

    @property
    def files_to_remove(self):
        return sorted(self.file_names - self._source_docs.file_names)

    @property
    def files_to_update(self):
        return [
            file_name
            for file_name in sorted(self._source_docs.file_names & self.file_names)
            if self._need_update(file_name)
        ]

    @property
    def files_unchanged(self):
        return [
            file_name
            for file_name in sorted(self._source_docs.file_names & self.file_names)
            if not self._need_update(file_name)
        ]


class TranslationPromptBuilder:
    def __init__(self, lang_tag: str, lang_prompt_path: Path, prompt_template_dir: Path) -> None:
        self._lang_tag = lang_tag
        self._lang_name = LANG_TAG2NAME[lang_tag]
        self._lang_prompt = read_text(lang_prompt_path).strip()
        env = Environment(
            loader=FileSystemLoader(prompt_template_dir),
            undefined=StrictUndefined,
            autoescape=False,  # nosec B701: the rendered prompt is MDX, HTML escaping would corrupt it
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self._prompt_template = env.get_template(TRANSLATION_PROMPT_TEMPLATE_NAME)

    def _build_title_prompt(self, task_prompt: str, doc_en_title: str) -> str:
        return (
            "# Task\n\n"
            f"Translate a LeRobot documentation title from English into {self._lang_name} "
            f"(language tag: `{self._lang_tag}`).\n"
            f"{task_prompt}\n\n"
            "# Output format\n\n"
            "Return one single line holding the translated title and nothing else.\n\n"
            f"{self._lang_prompt}\n\n"
            "# Source English Title\n\n"
            f"{doc_en_title.strip()}\n"
        )

    def build_content_translation_prompt(self, doc_en_content: str) -> str:
        task_prompt = "This page has no translation yet. Translate the whole English source below."
        prompt = self._prompt_template.render(
            task_prompt=task_prompt,
            lang_prompt=self._lang_prompt,
            doc_source_content=doc_en_content,
        )
        return prompt

    def build_content_update_prompt(
        self, doc_en_content: str, doc_en_diff: str, doc_lang_content_old: str
    ) -> str:
        task_prompt = (
            "This page already has a translation, but the English source changed afterwards. The diff "
            "below goes from the version that translation was made from to the current English source. "
            "Reuse the old translation wherever the English is unchanged, retranslate only the passages "
            "the diff touches, and return the complete updated page, never a diff or a fragment."
        )
        prompt = self._prompt_template.render(
            task_prompt=task_prompt,
            lang_prompt=self._lang_prompt,
            doc_source_content=doc_en_content,
            doc_source_diff=doc_en_diff,
            doc_target_content=doc_lang_content_old,
        )
        return prompt

    def build_doc_title_translation_prompt(self, doc_en_title: str) -> str:
        task_prompt = (
            "The title labels a single page in the documentation sidebar. Keep it short, and keep "
            "product names, file names and acronyms in English."
        )
        return self._build_title_prompt(task_prompt, doc_en_title)

    def build_section_title_translation_prompt(self, doc_en_section_title: str) -> str:
        task_prompt = (
            "The title labels a group of pages in the documentation sidebar. Keep it short, keep product "
            "names and acronyms in English, and prefer the most conventional wording, because every page "
            "of the group is filed under it."
        )
        return self._build_title_prompt(task_prompt, doc_en_section_title)


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str


class Translator:
    def __init__(self, prompt_builder: TranslationPromptBuilder, llm_config: LlmConfig) -> None:
        self._prompt_builder = prompt_builder
        self._llm_config = llm_config
        self._inference_client = InferenceClient(base_url=llm_config.base_url, api_key=llm_config.api_key)

    def _llm_inference(self, prompt: str) -> str:
        response = self._inference_client.chat.completions.create(
            model=self._llm_config.model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""

    def _llm_inference_title(self, prompt: str, doc_en_title: str) -> str:
        response = self._llm_inference(prompt)
        title_lines = [line.strip() for line in response.splitlines() if line.strip()]
        return title_lines[0] if title_lines else doc_en_title

    def translate_from_scratch(self, doc_en_content: str) -> str:
        prompt = self._prompt_builder.build_content_translation_prompt(doc_en_content)
        return self._llm_inference(prompt)

    def translate_title(self, doc_en_title: str) -> str:
        prompt = self._prompt_builder.build_doc_title_translation_prompt(doc_en_title)
        return self._llm_inference_title(prompt, doc_en_title)

    def translate_section_title(self, doc_en_section_title: str) -> str:
        prompt = self._prompt_builder.build_section_title_translation_prompt(doc_en_section_title)
        return self._llm_inference_title(prompt, doc_en_section_title)

    def update_translation(self, doc_en_content: str, doc_en_diff: str, doc_lang_content_old: str) -> str:
        prompt = self._prompt_builder.build_content_update_prompt(
            doc_en_content, doc_en_diff, doc_lang_content_old
        )
        return self._llm_inference(prompt)


class TranslationChecker:
    def _check_line_count(self, en_content: str, translation_result: str) -> str | None:
        diff = abs(len(en_content.splitlines()) - len(translation_result.splitlines()))
        if diff <= LINE_COUNT_TOLERANCE:
            return None
        return f"line count differs by {diff} (tolerance {LINE_COUNT_TOLERANCE})"

    def _check_blank_line_count(self, en_content: str, translation_result: str) -> str | None:
        def count_blank_lines(text: str) -> int:
            return sum(1 for line in text.splitlines() if not line.strip())

        diff = abs(count_blank_lines(en_content) - count_blank_lines(translation_result))
        if diff <= BLANK_LINE_COUNT_TOLERANCE:
            return None
        return f"blank line count differs by {diff} (tolerance {BLANK_LINE_COUNT_TOLERANCE})"

    def check_all(self, en_content: str, translation_result: str) -> list[str]:
        failures = (
            self._check_line_count(en_content, translation_result),
            self._check_blank_line_count(en_content, translation_result),
        )
        return [failure for failure in failures if failure is not None]


class DocAction(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    REMOVED = "removed"
    SKIPPED = "skipped"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class DocOutcome:
    file_name: str
    action: DocAction
    attempts: int = 0
    error: str | None = None
    check_failures: list[str] = field(default_factory=list)
    diff_stat: tuple[int, int] | None = None
    commits_since: int = 0
    translated_from: Commit | None = None
    english_now: Commit | None = None
    source_diff: str | None = None


@dataclass(frozen=True)
class TranslationReport:
    lang_tag: str
    lang_name: str
    model: str
    source: Commit | None
    last_sync: Commit | None
    docs: list[DocOutcome]

    def save(self, report_path: Path) -> None:
        write_text(report_path, json.dumps(asdict(self), indent=2, ensure_ascii=False))


class TranslationPipeline:
    def __init__(
        self,
        source_docs: SourceDocs,
        target_docs: TargetDocs,
        translator: Translator,
        translation_checker: TranslationChecker,
        max_attempts: int,
        selected_file_names: set[str] | None = None,
    ) -> None:
        self._source_docs = source_docs
        self._target_docs = target_docs
        self._translator = translator
        self._translation_checker = translation_checker
        self._max_attempts = max_attempts
        self._selected_file_names = selected_file_names

    def _select(self, file_names: list[str]) -> list[str]:
        if self._selected_file_names is None:
            return file_names
        return [file_name for file_name in file_names if file_name in self._selected_file_names]

    def _errored_doc(self, file_name: str, error: Exception) -> DocOutcome:
        translated_from = self._target_docs.get_translated_commit(file_name)
        return DocOutcome(
            file_name,
            DocAction.SKIPPED,
            error=repr(error),
            commits_since=self._source_docs.count_commits_since(file_name, translated_from),
            translated_from=translated_from,
            english_now=self._source_docs.get_current_commit(file_name),
        )

    def _run_docs(
        self, file_names: list[str], desc: str, run_doc: Callable[[str], DocOutcome]
    ) -> list[DocOutcome]:
        outcomes: list[DocOutcome] = []
        for index, file_name in enumerate(file_names, start=1):
            print(f"{desc} [{index}/{len(file_names)}] {file_name}", file=sys.stderr)
            started = time.monotonic()
            try:
                outcome = run_doc(file_name)
            except Exception as error:  # one page must never take the whole run down
                print(traceback.format_exc(), end="", file=sys.stderr)
                outcome = self._errored_doc(file_name, error)
            attempts = f" ({outcome.attempts} attempts)" if outcome.attempts else ""
            print(f"  {outcome.action} in {time.monotonic() - started:.0f}s{attempts}", file=sys.stderr)
            outcomes.append(outcome)
        return outcomes

    def _remove_doc(self, file_name: str) -> DocOutcome:
        translated_from = self._target_docs.get_translated_commit(file_name)
        english_now = self._source_docs.get_current_commit(file_name)
        self._target_docs.remove_doc(file_name)
        return DocOutcome(
            file_name, DocAction.REMOVED, translated_from=translated_from, english_now=english_now
        )

    def _add_doc(self, file_name: str) -> DocOutcome:
        source_doc_content = self._source_docs.get_doc_content(file_name)
        source_doc_titles = self._source_docs.get_section_and_doc_title(file_name)
        if source_doc_titles is None:
            raise ValueError(f"{file_name} is not listed in {TOCTREE_FILE_NAME}")
        source_doc_section_title, source_doc_title = source_doc_titles

        translated_doc_title = self._translator.translate_title(source_doc_title)
        if self._target_docs.get_section_title(source_doc_section_title) is None:
            self._target_docs.set_section_title(
                source_doc_section_title,
                self._translator.translate_section_title(source_doc_section_title),
            )

        check_failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            translated_content = self._translator.translate_from_scratch(source_doc_content)
            check_failures = self._translation_checker.check_all(source_doc_content, translated_content)
            if not check_failures:
                self._target_docs.add_doc(file_name, translated_content, translated_doc_title)
                action, attempts = DocAction.ADDED, attempt
                break
        else:
            action, attempts = DocAction.SKIPPED, self._max_attempts

        return DocOutcome(
            file_name,
            action,
            attempts=attempts,
            check_failures=check_failures,
            english_now=self._source_docs.get_current_commit(file_name),
        )

    def _update_doc(self, file_name: str) -> DocOutcome:
        source_doc_content = self._source_docs.get_doc_content(file_name)
        doc_lang_content_old = self._target_docs.get_doc_content(file_name)
        old_version = self._target_docs.get_last_translated_version(file_name)
        source_doc_diff = None if old_version is None else self._source_docs.get_diff(file_name, old_version)
        translated_from = self._target_docs.get_translated_commit(file_name)

        check_failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            if source_doc_diff is None:
                updated_content = self._translator.translate_from_scratch(source_doc_content)
            else:
                updated_content = self._translator.update_translation(
                    source_doc_content, source_doc_diff, doc_lang_content_old
                )
            check_failures = self._translation_checker.check_all(source_doc_content, updated_content)
            if not check_failures:
                self._target_docs.update_doc(file_name, updated_content)
                action, attempts = DocAction.UPDATED, attempt
                break
        else:
            action, attempts = DocAction.SKIPPED, self._max_attempts

        diff_stat = None if old_version is None else self._source_docs.get_diff_stat(file_name, old_version)
        return DocOutcome(
            file_name,
            action,
            attempts=attempts,
            check_failures=check_failures,
            diff_stat=diff_stat,
            commits_since=self._source_docs.count_commits_since(file_name, translated_from),
            translated_from=translated_from,
            english_now=self._source_docs.get_current_commit(file_name),
            source_diff=source_doc_diff,
        )

    def remove_docs(self) -> list[DocOutcome]:
        return self._run_docs(self._select(self._target_docs.files_to_remove), "remove", self._remove_doc)

    def add_docs(self) -> list[DocOutcome]:
        return self._run_docs(self._select(self._target_docs.files_to_add), "add", self._add_doc)

    def update_docs(self) -> list[DocOutcome]:
        return self._run_docs(self._select(self._target_docs.files_to_update), "update", self._update_doc)

    def run(self) -> list[DocOutcome]:
        # taken first: adding or updating a doc makes it look unchanged afterwards
        unchanged = [
            DocOutcome(file_name, DocAction.UNCHANGED)
            for file_name in self._select(self._target_docs.files_unchanged)
        ]
        return self.remove_docs() + self.add_docs() + self.update_docs() + unchanged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-tag", choices=LANG_TAG2NAME, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument(
        "--select", nargs="+", metavar="FILE_NAME", help="doc file names to process, defaults to all"
    )
    parser.add_argument("--report", type=Path, help="write the run report as JSON to this path")

    args = parser.parse_args()
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        parser.error(f"set {API_KEY_ENV} to the API key for --base_url")
    lang_tag = args.lang_tag
    selected_file_names = set(args.select) if args.select else None

    repo_root = Path(__file__).resolve().parent.parent.parent

    doc_root = repo_root / "docs" / "source"
    doc_lang_dir = doc_root / lang_tag
    lang_prompt_path = repo_root / f"docs/translation-prompts/{lang_tag}.md"

    git = GitRepo(repo_root)
    source_docs = SourceDocs(doc_root, git)
    localized_docs = TargetDocs(doc_lang_dir, source_docs)

    llm_config = LlmConfig(base_url=args.base_url, api_key=api_key, model=args.model)
    prompt_builder = TranslationPromptBuilder(lang_tag, lang_prompt_path, TRANSLATION_PROMPT_TEMPLATE_DIR)
    translator = Translator(prompt_builder, llm_config)
    translation_checker = TranslationChecker()
    translation_pipeline = TranslationPipeline(
        source_docs, localized_docs, translator, translation_checker, 3, selected_file_names
    )
    outcomes = translation_pipeline.run()
    if args.report:
        TranslationReport(
            lang_tag=lang_tag,
            lang_name=LANG_TAG2NAME[lang_tag],
            model=args.model,
            source=localized_docs.get_latest_source_commit(),
            last_sync=localized_docs.get_last_sync_commit(),
            docs=outcomes,
        ).save(args.report)


if __name__ == "__main__":
    main()
