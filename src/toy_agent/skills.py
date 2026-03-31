"""Skills loader: discover and load skills from user + project level skills dirs.

Each skill is a .md file with YAML frontmatter:
---
description: shown to LLM for deciding when to invoke this skill
---

Skill prompt content goes here.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    content: str  # prompt body after frontmatter


def load_skills(project_dir: str | Path = ".", home: Path | None = None) -> list[Skill]:
    """Load and merge skills from user-level and project-level skills dirs.

    Project-level overrides user-level on name conflict.
    """
    resolved = Path(project_dir).resolve()
    home = home or Path.home()

    user_dir = home / ".toy-agent" / "skills"
    project_skills_dir = resolved / ".toy-agent" / "skills"

    skills: dict[str, Skill] = {}

    for skill in _load_skills_dir(user_dir):
        skills[skill.name] = skill

    for skill in _load_skills_dir(project_skills_dir):
        skills[skill.name] = skill

    return list(skills.values())


def get_skill(skills: list[Skill], name: str) -> Skill | None:
    """Find a skill by name. Returns None if not found."""
    for skill in skills:
        if skill.name == name:
            return skill
    return None


def _load_skills_dir(dir_path: Path) -> list[Skill]:
    """Load all skill .md files from a directory."""
    if not dir_path.is_dir():
        return []

    skills = []
    for file in sorted(dir_path.glob("**/*.md")):
        if file.name.lower() != "skill.md":
            continue
        skill = _parse_skill(file)
        if skill:
            skills.append(skill)
    return skills


def _parse_skill(file: Path) -> Skill | None:
    """Parse a single skill .md file."""
    try:
        content = file.read_text()
    except OSError:
        return None

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        print(f"[skills] {file.name}: missing or invalid frontmatter")
        return None

    frontmatter_raw, body = match.groups()
    meta = _parse_frontmatter(frontmatter_raw)

    name = meta.get("name", file.parent.name or "skill")
    description = meta.get("description", "")

    return Skill(name=name, description=description, content=body.strip())


def _parse_frontmatter(raw: str) -> dict[str, str]:
    """Parse simple YAML frontmatter (key: value pairs only)."""
    result = {}
    for line in raw.splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result
