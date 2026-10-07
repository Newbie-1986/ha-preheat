"""Regression tests for upcoming preheat starts from HA schedule helpers."""

# Covers Ecronika/ha-preheat#5.

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.preheat.const import CONF_SCHEDULE_ENTITY, REASON_OFF
from custom_components.preheat.coordinator import PreheatingCoordinator
from custom_components.preheat.providers import ProviderDecision, ScheduleProvider


def _make_schedule_provider(state: str, next_event: str | None):
    hass = MagicMock()
    entry = MagicMock()
    entry.options = {CONF_SCHEDULE_ENTITY: "schedule.test_comfort"}
    entry.data = {}

    schedule_state = MagicMock()
    schedule_state.state = state
    schedule_state.attributes = {}
    if next_event is not None:
        schedule_state.attributes["next_event"] = next_event

    hass.states.get.return_value = schedule_state
    provider = ScheduleProvider(hass, entry, MagicMock())
    return provider, hass


def test_schedule_off_future_next_event_is_valid_start_source():
    """OFF + future next_event should represent the next comfort-session start."""
    now = datetime(2026, 10, 7, 20, 14, tzinfo=timezone.utc)
    future = now + timedelta(hours=10)

    provider, _ = _make_schedule_provider("off", future.isoformat())

    with patch(
        "custom_components.preheat.providers.dt_util.parse_datetime",
        return_value=future,
    ):
        assert provider.get_next_session_start(now) == future

        decision = provider.get_decision({"now": now})

    assert decision.is_valid is True
    assert decision.is_shadow is False
    assert decision.session_end is None
    assert decision.invalid_reason is None


def test_schedule_on_keeps_existing_session_end_behavior():
    """ON schedules must keep treating next_event as the current session end."""
    now = datetime(2026, 10, 8, 6, 0, tzinfo=timezone.utc)
    session_end = now + timedelta(hours=1)

    provider, _ = _make_schedule_provider("on", session_end.isoformat())

    assert provider.get_next_session_start(now) is None

    with patch(
        "custom_components.preheat.optimal_stop.dt_util.parse_datetime",
        return_value=session_end,
    ):
        decision = provider.get_decision({"now": now})

    assert decision.is_valid is True
    assert decision.session_end == session_end
    assert decision.invalid_reason is None


def test_schedule_off_without_valid_next_event_keeps_fallback_behavior():
    """OFF without a future event must remain invalid so House/Learned can win."""
    now = datetime(2026, 10, 7, 20, 14, tzinfo=timezone.utc)
    provider, _ = _make_schedule_provider("off", None)

    assert provider.get_next_session_start(now) is None

    decision = provider.get_decision({"now": now})

    assert decision.is_valid is False
    assert decision.session_end is None
    assert decision.invalid_reason == REASON_OFF


@pytest.mark.asyncio
async def test_collect_context_prioritizes_upcoming_schedule_start():
    """Explicit schedule start must outrank House/Learned arrival predictions."""
    now = datetime(2026, 10, 7, 20, 14, tzinfo=timezone.utc)
    schedule_start = now + timedelta(hours=10)
    learned_start = now + timedelta(hours=11)

    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.title = "Test"
    entry.options = {}
    entry.data = {}

    with patch.object(PreheatingCoordinator, "_setup_listeners"):
        coordinator = PreheatingCoordinator(hass, entry)

    coordinator._get_operative_temperature = AsyncMock(return_value=19.0)
    coordinator._get_valve_position_with_fallback = MagicMock(return_value=None)
    coordinator.session_manager.check_debounce = AsyncMock()
    coordinator._track_temperature_gradient = MagicMock()
    coordinator.history_buffer.append = MagicMock()
    coordinator._get_conf = MagicMock(return_value=None)
    coordinator._get_blocked_dates_from_calendar = AsyncMock(return_value=set())
    coordinator._get_allowed_weekdays = MagicMock(return_value=None)
    coordinator.planner.get_next_scheduled_event = MagicMock(return_value=learned_start)
    coordinator.schedule_provider.get_next_session_start = MagicMock(
        return_value=schedule_start
    )
    coordinator._get_outdoor_temp_current = AsyncMock(return_value=5.0)
    coordinator._update_outdoor_availability_issue = MagicMock()
    coordinator._get_target_setpoint = AsyncMock(return_value=21.0)
    coordinator.weather_service = None
    coordinator.house_collector = None

    with (
        patch("custom_components.preheat.coordinator.dt_util.now", return_value=now),
        patch("custom_components.preheat.coordinator.dt_util.utcnow", return_value=now),
    ):
        context = await coordinator._collect_context()

    assert context["next_event"] == schedule_start


def test_future_preheat_start_is_exposed_before_trigger_time():
    """The next-start sensor should get event - predicted duration in advance."""
    now = datetime(2026, 10, 7, 20, 14, tzinfo=timezone.utc)
    event = now + timedelta(hours=10)
    predicted_duration = 30.0

    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.title = "Test"
    entry.options = {}
    entry.data = {}

    with patch.object(PreheatingCoordinator, "_setup_listeners"):
        coordinator = PreheatingCoordinator(hass, entry)

    coordinator._window_open_detected = False
    coordinator.hold_active = False
    coordinator.enable_active = True
    coordinator._get_conf = MagicMock(return_value=False)

    final_decision = ProviderDecision(
        should_stop=False,
        session_end=None,
        is_valid=True,
        is_shadow=False,
    )

    (
        should_start,
        start_time,
        _effective_departure,
        blocked,
        _blocked_reasons,
    ) = coordinator._apply_blocks_and_suppression(
        ctx={"next_event": event},
        pred={"predicted_duration": predicted_duration},
        now=now,
        frost_override=False,
        final_decision=final_decision,
        is_optimal_stop_active=False,
        should_start=False,
        start_time=None,
    )

    assert should_start is False
    assert blocked is False
    assert start_time == event - timedelta(minutes=predicted_duration)
