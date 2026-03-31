"""Unit tests for skills loader."""

from pathlib import Path

from src.toy_agent.skills import _parse_frontmatter, _parse_skill, load_skills


class TestParseSkill:
    def test_valid_skill(self, tmp_path: Path):
        file = tmp_path / "code-review" / "SKILL.md"
        file.parent.mkdir()
        file.write_text(
            "---\n"
            "name: code-review\n"
            "description: Expert at reviewing code\n"
            "---\n"
            "You are a code reviewer.\n"
            "Check for bugs.\n"
        )
        skill = _parse_skill(file)

        assert skill is not None
        assert skill.name == "code-review"
        assert skill.description == "Expert at reviewing code"
        assert skill.content == "You are a code reviewer.\nCheck for bugs."

    def test_name_defaults_to_parent_dir(self, tmp_path: Path):
        file = tmp_path / "my-skill" / "SKILL.md"
        file.parent.mkdir()
        file.write_text("---\ndescription: A skill without name in frontmatter\n---\nPrompt body.\n")
        skill = _parse_skill(file)

        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "A skill without name in frontmatter"
        assert skill.content == "Prompt body."

    def test_missing_frontmatter(self, tmp_path: Path):
        file = tmp_path / "bad" / "SKILL.md"
        file.parent.mkdir()
        file.write_text("Just plain content.\n")

        skill = _parse_skill(file)

        assert skill is None

    def test_empty_body(self, tmp_path: Path):
        file = tmp_path / "empty" / "SKILL.md"
        file.parent.mkdir()
        file.write_text("---\ndescription: body is empty\n---\n")
        skill = _parse_skill(file)

        assert skill is not None
        assert skill.content == ""


class TestParseFrontmatter:
    def test_single_line(self):
        result = _parse_frontmatter("name: code-review\ndescription: Expert")
        assert result["name"] == "code-review"
        assert result["description"] == "Expert"

    def test_empty_value(self):
        result = _parse_frontmatter("name: test\nunknown:\ntype: string")
        assert result["name"] == "test"
        assert result["unknown"] == ""
        assert result["type"] == "string"


class TestLoadSkillsDir:
    def test_only_loads_skill_md(self, tmp_path: Path):
        (tmp_path / "code-review").mkdir()
        (tmp_path / "code-review" / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: code review\n---\nReview code."
        )
        (tmp_path / "readme.md").write_text("Not a skill")
        (tmp_path / "other.md").write_text("---\nname: other\n---\nX")

        from src.toy_agent.skills import _load_skills_dir

        skills = _load_skills_dir(tmp_path)

        assert len(skills) == 1
        assert skills[0].name == "code-review"


class TestLoadSkills:
    def test_project_overrides_user(self, tmp_path: Path):
        user_root = tmp_path / "user_home" / ".toy-agent" / "skills"
        project_root = tmp_path / "project" / ".toy-agent" / "skills"
        user_root.mkdir(parents=True)
        project_root.mkdir(parents=True)

        (user_root / "shared").mkdir()
        (user_root / "shared" / "SKILL.md").write_text("---\nname: shared\ndescription: from user\n---\nUser content.")
        (user_root / "user-only").mkdir()
        (user_root / "user-only" / "SKILL.md").write_text("---\nname: user-only\ndescription: u\n---\nU.")

        (project_root / "shared").mkdir()
        (project_root / "shared" / "SKILL.md").write_text(
            "---\nname: shared\ndescription: from project\n---\nProject content."
        )
        (project_root / "project-only").mkdir()
        (project_root / "project-only" / "SKILL.md").write_text("---\nname: project-only\ndescription: p\n---\nP.")

        skills = load_skills(tmp_path / "project", home=tmp_path / "user_home")

        skill_map = {s.name: s for s in skills}

        # shared overridden by project
        assert skill_map["shared"].description == "from project"
        assert skill_map["shared"].content == "Project content."
        # project-only only in project
        assert "project-only" in skill_map
        # user-only loaded from user level (no override from project)
        assert "user-only" in skill_map
        assert len(skills) == 3

    def test_no_dirs_returns_empty(self, tmp_path: Path):
        skills = load_skills(tmp_path)
        assert skills == []
