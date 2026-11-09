"""AVT-1..4: the voice page contains the avatar, its hooks, and keeps the old orb as the fallback."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "frontend" / "voice_ui.html"


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_avatar_and_old_orb_are_both_present(html):
    assert 'id="avatar"' in html
    assert 'id="orb"' in html and 'onclick="startSystem()"' in html
    assert 'avatar-toggle' not in html  # no on/off button: avatar is always shown


def test_ids_are_unique(html):
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids))


def test_all_four_states_exist(html):
    for state in ("idle", "listening", "thinking", "speaking"):
        assert f"av-{state}" in html


def test_both_audio_paths_tell_the_avatar(html):
    # answer from /speak and chart explanation from /analyze_csv
    assert html.count("avatarSpeak(currentAudio)") == 2
    assert "window.avatarSpeak" in html


def test_every_play_call_is_preceded_by_the_hook(html):
    for match in re.finditer(r"currentAudio\.play\(\);", html):
        assert "avatarSpeak(currentAudio);" in html[max(0, match.start() - 120):match.start()]


def test_fallback_and_reduced_motion(html):
    assert "analysing" in html and "fallback rhythm" in html
    assert "prefers-reduced-motion" in html


def test_avatar_is_keyboard_reachable(html):
    assert 'role="button"' in html and 'tabindex="0"' in html
    assert "aria-label" in html
