from __future__ import annotations

import unittest

import pytest

pytest.importorskip("websockets", reason="the meow watchdog needs the websockets client")

from scripts.cat_projector_meow_watchdog import (  # noqa: E402
    DEFAULT_BLOCKED_BROAD_LABELS,
    should_accept_meow_event,
    should_start_projector,
)

#: The gate's own fixture. DEFAULT_SOURCE_IDS is the deployment roster and is
#: empty without a site config, so reading it here made these tests assert on
#: whatever the installing host happened to configure.
SOURCE_IDS = {"example_room_webcam"}


class CatProjectorMeowWatchdogTest(unittest.TestCase):
    def _accepted_payload(self) -> dict[str, object]:
        return {
            "source_id": "example_room_webcam",
            "meow_score": 0.84,
            "broad_acoustic_label": "cat_meow",
            "broad_acoustic_confidence": 0.42,
        }

    def test_accepts_living_room_meow_candidate(self) -> None:
        decision = should_accept_meow_event(
            self._accepted_payload(),
            source_ids=SOURCE_IDS,
            min_score=0.62,
            blocked_broad_labels=DEFAULT_BLOCKED_BROAD_LABELS,
            broad_block_confidence=0.25,
        )

        self.assertTrue(decision.accepted)

    def test_rejects_confident_non_cat_broad_label(self) -> None:
        decision = should_accept_meow_event(
            {
                "source_id": "example_room_webcam",
                "meow_score": 0.84,
                "broad_acoustic_label": "speech",
                "broad_acoustic_confidence": 0.71,
            },
            source_ids=SOURCE_IDS,
            min_score=0.62,
            blocked_broad_labels=DEFAULT_BLOCKED_BROAD_LABELS,
            broad_block_confidence=0.25,
        )

        self.assertFalse(decision.accepted)
        self.assertIn("blocked_broad_label:speech", decision.reason)

    def test_rejects_non_play_intent_when_play_probability_is_low(self) -> None:
        payload = self._accepted_payload()
        payload["intent_hint"] = {
            "label": "complaint",
            "confidence": 0.41,
            "class_probabilities": {"play": 0.04},
        }

        decision = should_accept_meow_event(
            payload,
            source_ids=SOURCE_IDS,
            min_score=0.62,
            blocked_broad_labels=DEFAULT_BLOCKED_BROAD_LABELS,
            broad_block_confidence=0.25,
        )

        self.assertFalse(decision.accepted)

    def test_accepts_play_intent(self) -> None:
        payload = self._accepted_payload()
        payload["intent_hint"] = {
            "label": "play",
            "confidence": 0.82,
            "class_probabilities": {"play": 0.79},
        }

        decision = should_accept_meow_event(
            payload,
            source_ids=SOURCE_IDS,
            min_score=0.62,
            blocked_broad_labels=DEFAULT_BLOCKED_BROAD_LABELS,
            broad_block_confidence=0.25,
        )

        self.assertTrue(decision.accepted)

    def test_accepts_legacy_payload_without_intent_hint(self) -> None:
        decision = should_accept_meow_event(
            self._accepted_payload(),
            source_ids=SOURCE_IDS,
            min_score=0.62,
            blocked_broad_labels=DEFAULT_BLOCKED_BROAD_LABELS,
            broad_block_confidence=0.25,
        )

        self.assertTrue(decision.accepted)

    def test_retries_failed_session_when_renderer_is_unavailable(self) -> None:
        decision = should_start_projector(session_state="on", renderer_state="unavailable")

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "session_on_renderer_unavailable")

    def test_does_not_restart_current_playback(self) -> None:
        decision = should_start_projector(session_state="on", renderer_state="playing")

        self.assertFalse(decision.accepted)


if __name__ == "__main__":
    unittest.main()
