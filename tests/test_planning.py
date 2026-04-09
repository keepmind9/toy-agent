"""Unit tests for Planning feature."""

import json
from unittest.mock import MagicMock

import pytest

from toy_agent.agent import Agent
from toy_agent.planning import Plan, PlanHook, PlanStep, ReActPlanHook


class TestDataclasses:
    def test_plan_step_defaults(self):
        """PlanStep has sensible defaults."""
        step = PlanStep(id=1, description="Do something")
        assert step.status == "pending"

    def test_plan_step_custom_status(self):
        """PlanStep status can be set."""
        step = PlanStep(id=1, description="Done step", status="done")
        assert step.status == "done"

    def test_plan_creation(self):
        """Plan holds goal and steps."""
        steps = [PlanStep(id=1, description="Step 1"), PlanStep(id=2, description="Step 2")]
        plan = Plan(goal="Test goal", steps=steps)
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 2


class TestPlanHookCheckNeedsPlan:
    @pytest.mark.anyio
    async def test_simple_question_no_plan(self):
        """Simple questions don't trigger planning."""
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps({"needs_plan": False})
        client.chat.completions.create.return_value = response

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=True)
        result = await hook._check_needs_plan("What time is it?")
        assert result is False

    @pytest.mark.anyio
    async def test_complex_task_needs_plan(self):
        """Complex tasks trigger planning."""
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps({"needs_plan": True})
        client.chat.completions.create.return_value = response

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=True)
        result = await hook._check_needs_plan("Analyze the entire codebase and suggest improvements")
        assert result is True

    @pytest.mark.anyio
    async def test_api_error_returns_false(self):
        """If classification API fails, don't plan."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=True)
        result = await hook._check_needs_plan("Some task")
        assert result is False


class TestPlanHookGeneratePlan:
    @pytest.mark.anyio
    async def test_generate_plan_returns_steps(self):
        """Plan generation returns a Plan with steps."""
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(
            {
                "goal": "Analyze code",
                "steps": ["Scan structure", "Check quality", "Summarize"],
            }
        )
        client.chat.completions.create.return_value = response

        hook = PlanHook(client=client, model="gpt-4o-mini")
        plan = await hook._generate_plan("Analyze the code", ["read_file", "run_bash"])

        assert plan is not None
        assert plan.goal == "Analyze code"
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Scan structure"
        assert plan.steps[0].id == 1

    @pytest.mark.anyio
    async def test_generate_plan_api_error_returns_none(self):
        """If plan generation API fails, return None."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")

        hook = PlanHook(client=client, model="gpt-4o-mini")
        plan = await hook._generate_plan("Do something", [])
        assert plan is None


class TestPlanHookFormatPlan:
    def test_format_plan(self):
        """Plan formatting produces readable text."""
        plan = Plan(
            goal="Test goal",
            steps=[
                PlanStep(id=1, description="Step one"),
                PlanStep(id=2, description="Step two"),
            ],
        )
        hook = PlanHook(client=MagicMock(), model="gpt-4o-mini")
        text = hook._format_plan(plan)

        assert "Test goal" in text
        assert "1. Step one" in text
        assert "2. Step two" in text


class TestAgentPlanIntegration:
    @pytest.mark.anyio
    async def test_plan_true_generates_plan(self):
        """plan=True forces plan generation without auto-detect."""
        client = MagicMock()

        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = json.dumps(
            {
                "goal": "Analyze code",
                "steps": ["Scan structure", "Check quality"],
            }
        )

        text_msg = MagicMock()
        text_msg.content = "Done analyzing"
        text_msg.tool_calls = None

        client.chat.completions.create.side_effect = [
            plan_response,
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=False)
        agent = Agent(client=client, hooks=[hook])
        result = await agent.run("Analyze the code", plan=True)

        assert result == "Done analyzing"
        assert any("Goal: Analyze code" in m.get("content", "") for m in agent.messages)

    @pytest.mark.anyio
    async def test_plan_false_skips_planning(self):
        """plan=False skips planning even when auto=True."""
        client = MagicMock()
        text_msg = MagicMock()
        text_msg.content = "Quick answer"
        text_msg.tool_calls = None
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=text_msg)])

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=True)
        agent = Agent(client=client, hooks=[hook])
        result = await agent.run("Hello", plan=False)

        assert result == "Quick answer"
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.anyio
    async def test_no_plan_hook_works_as_before(self):
        """Agent without PlanHook works normally with plan param."""
        client = MagicMock()
        text_msg = MagicMock()
        text_msg.content = "normal response"
        text_msg.tool_calls = None
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=text_msg)])

        agent = Agent(client=client)
        result = await agent.run("Hello", plan=True)

        assert result == "normal response"

    @pytest.mark.anyio
    async def test_auto_detect_triggers_plan(self):
        """Auto mode: LLM says task needs plan -> plan generated."""
        client = MagicMock()

        classify_response = MagicMock()
        classify_response.choices = [MagicMock()]
        classify_response.choices[0].message.content = json.dumps({"needs_plan": True})

        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = json.dumps(
            {
                "goal": "Complex analysis",
                "steps": ["Step 1", "Step 2"],
            }
        )

        text_msg = MagicMock()
        text_msg.content = "Analysis complete"
        text_msg.tool_calls = None

        client.chat.completions.create.side_effect = [
            classify_response,
            plan_response,
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=True)
        agent = Agent(client=client, hooks=[hook])
        result = await agent.run("Do a complex analysis")

        assert result == "Analysis complete"
        assert any("Goal: Complex analysis" in m.get("content", "") for m in agent.messages)

    @pytest.mark.anyio
    async def test_auto_detect_skips_simple_task(self):
        """Auto mode: LLM says task doesn't need plan -> no plan generated."""
        client = MagicMock()

        classify_response = MagicMock()
        classify_response.choices = [MagicMock()]
        classify_response.choices[0].message.content = json.dumps({"needs_plan": False})

        text_msg = MagicMock()
        text_msg.content = "It's 3pm"
        text_msg.tool_calls = None

        client.chat.completions.create.side_effect = [
            classify_response,
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        hook = PlanHook(client=client, model="gpt-4o-mini", auto=True)
        agent = Agent(client=client, hooks=[hook])
        result = await agent.run("What time is it?")

        assert result == "It's 3pm"
        assert not any("Goal:" in m.get("content", "") for m in agent.messages if m.get("role") == "assistant")


class TestReActPlanHookOnBeforeLoop:
    @pytest.mark.anyio
    async def test_injects_system_hint_no_api_calls(self):
        """ReActPlanHook injects guidance prompt without any LLM API calls."""
        client = MagicMock()
        text_msg = MagicMock()
        text_msg.content = "Direct answer"
        text_msg.tool_calls = None
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=text_msg)])

        hook = ReActPlanHook()
        agent = Agent(client=client, hooks=[hook])
        await agent.run("Hello")

        # Only 1 API call (the main agent loop), no extra planning calls
        assert client.chat.completions.create.call_count == 1
        # System hint injected as assistant message
        assert any("complex task" in m.get("content", "") for m in agent.messages if m.get("role") == "assistant")

    @pytest.mark.anyio
    async def test_stores_agent_reference(self):
        """ReActPlanHook stores agent reference after on_before_loop."""
        client = MagicMock()
        text_msg = MagicMock()
        text_msg.content = "Done"
        text_msg.tool_calls = None
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=text_msg)])

        hook = ReActPlanHook()
        agent = Agent(client=client, hooks=[hook])
        await agent.run("Hello")

        assert hook._agent is agent


class TestReActPlanHookPlanParsing:
    def test_parses_plan_declaration(self):
        """on_message parses a plan with Goal + numbered steps."""
        hook = ReActPlanHook()
        hook._agent = MagicMock()

        hook.on_message(
            role="assistant",
            content="Let me plan this.\nGoal: Analyze code\n\n1. Scan structure\n2. Check quality\n3. Summarize",
        )

        assert hook._plan is not None
        assert hook._plan.goal == "Analyze code"
        assert len(hook._plan.steps) == 3
        assert hook._plan.steps[0].description == "Scan structure"
        hook._agent._emit.assert_called_once_with("on_plan", plan=hook._plan)

    def test_no_plan_without_goal(self):
        """on_message does not parse a plan without Goal line."""
        hook = ReActPlanHook()
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="1. Do this\n2. Do that")

        assert hook._plan is None

    def test_no_plan_without_numbered_steps(self):
        """on_message does not parse a plan without numbered steps."""
        hook = ReActPlanHook()
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="Goal: Something\nJust a thought")

        assert hook._plan is None


class TestReActPlanHookStepDetection:
    def test_detects_step_complete_marker(self):
        """on_message detects [Step X complete] and updates step status."""
        hook = ReActPlanHook()
        hook._plan = Plan(
            goal="Test",
            steps=[PlanStep(id=1, description="First"), PlanStep(id=2, description="Second")],
        )
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="I did the first thing. [Step 1 complete]")

        assert hook._plan.steps[0].status == "done"
        assert hook._plan.steps[1].status == "pending"

    def test_detects_step_skipped_marker(self):
        """on_message detects [Step X skipped] and updates step status."""
        hook = ReActPlanHook()
        hook._plan = Plan(
            goal="Test",
            steps=[PlanStep(id=1, description="First"), PlanStep(id=2, description="Second")],
        )
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="Not needed. [Step 1 skipped]")

        assert hook._plan.steps[0].status == "skipped"

    def test_fires_plan_done_when_all_steps_complete(self):
        """on_plan_done fires when the last pending step is resolved."""
        hook = ReActPlanHook()
        hook._plan = Plan(
            goal="Test",
            steps=[PlanStep(id=1, description="First", status="done"), PlanStep(id=2, description="Second")],
        )
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="Finished. [Step 2 complete]")

        assert hook._plan.steps[1].status == "done"
        calls = hook._agent._emit.call_args_list
        assert any(c[0][0] == "on_plan_done" for c in calls)

    def test_ignores_non_assistant_messages(self):
        """on_message ignores user messages."""
        hook = ReActPlanHook()
        hook._plan = Plan(goal="Test", steps=[PlanStep(id=1, description="First")])
        hook._agent = MagicMock()

        hook.on_message(role="user", content="[Step 1 complete]")

        assert hook._plan.steps[0].status == "pending"
        hook._agent._emit.assert_not_called()

    def test_no_double_fire_for_same_step(self):
        """A step marker appearing twice only fires on_plan_step once."""
        hook = ReActPlanHook()
        hook._plan = Plan(goal="Test", steps=[PlanStep(id=1, description="First")])
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="[Step 1 complete] and again [Step 1 complete]")

        assert hook._plan.steps[0].status == "done"
        step_calls = [c for c in hook._agent._emit.call_args_list if c[0][0] == "on_plan_step"]
        assert len(step_calls) == 1

    def test_multiple_markers_in_one_message(self):
        """Multiple step markers in one message are all detected."""
        hook = ReActPlanHook()
        hook._plan = Plan(
            goal="Test",
            steps=[PlanStep(id=1, description="First"), PlanStep(id=2, description="Second")],
        )
        hook._agent = MagicMock()

        hook.on_message(role="assistant", content="[Step 1 complete] and [Step 2 skipped]")

        assert hook._plan.steps[0].status == "done"
        assert hook._plan.steps[1].status == "skipped"
        assert hook._agent._emit.call_count == 3


class TestReActPlanHookIntegration:
    @pytest.mark.anyio
    async def test_no_extra_api_calls(self):
        """ReActPlanHook makes zero extra LLM API calls."""
        client = MagicMock()
        text_msg = MagicMock()
        text_msg.content = "Direct answer"
        text_msg.tool_calls = None
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=text_msg)])

        hook = ReActPlanHook()
        agent = Agent(client=client, hooks=[hook])
        await agent.run("Hello")

        assert client.chat.completions.create.call_count == 1

    @pytest.mark.anyio
    async def test_model_self_plans_and_tracks(self):
        """End-to-end: model declares plan, reports steps, events fire."""
        client = MagicMock()

        text_msg = MagicMock()
        text_msg.content = (
            "Let me plan.\nGoal: Do things\n\n1. First thing\n2. Second thing\n\n"
            "Done first. [Step 1 complete]\nDone second. [Step 2 complete]"
        )
        text_msg.tool_calls = None

        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=text_msg)])

        hook = ReActPlanHook()
        mock_hook = MagicMock()
        agent = Agent(client=client, hooks=[hook, mock_hook])
        await agent.run("Do things")

        # Only 1 API call (main loop), no extra planning calls
        assert client.chat.completions.create.call_count == 1
        mock_hook.on_plan_step.assert_called()
        mock_hook.on_plan_done.assert_called_once()
