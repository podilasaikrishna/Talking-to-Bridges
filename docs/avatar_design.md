# Avatar: design, component and integration (AVT-1 to AVT-4)

## Choice
A simple animated 2D face drawn in SVG, inside `frontend/voice_ui.html`. No download, no internet and no extra library.
Why: the planner asks for a "feasible open-source/local" avatar. 3D talking heads (Three.js models, hosted services) need
large files, a network or a GPU, and can fail in a live demo. A 2D face is light, always works, and is enough to show the
four states and the speaking mouth.

## States
| State | When | What you see |
|---|---|---|
| idle | page open, agent not running | green ring, slow breathing, blinking |
| listening | microphone is open | red ring pulses, eyes a little wider, brows up |
| thinking | transcribing / waiting for the answer | blue ring turns, three dots, eyes look up-left |
| speaking | the answer audio is playing | blue ring pulses, mouth follows the voice |

## How it receives its input (the component's interface)
- **Page state:** the avatar watches the CSS class of the existing orb (`orb listening`, `orb thinking`, otherwise idle) with a
  `MutationObserver`. The existing voice code is not changed for this.
- **Audio:** `avatarSpeak(audioElement)` is called just before an answer is played (two places in `voice_ui.html`: the answer
  from `/speak`, and the chart explanation from `/analyze_csv`). The component reads the loudness of that audio with the
  Web Audio API (`AnalyserNode`), so the mouth opens with the voice. The mouth is also rounder for low sounds and wider for
  high sounds.
- `window.avatarState()` returns `{state, enabled, mouthOpen, analysing}` for debugging and tests.

## Fallback (AVT-4)
- If the audio analyser cannot be created, the mouth moves with a simple made-up rhythm while the state is "speaking".
- The avatar is always shown (no on/off button, team decision). The old orb stays hidden in the page because it drives the
  states. Clicking the avatar, or pressing Enter/Space on it, starts or pauses the agent.
- `prefers-reduced-motion` turns off breathing and ring animation.

## Limits
- The mouth follows loudness, not the actual words (no phoneme-level lip-sync).
- It is a cartoon face, not a person. Better lip-sync would need a viseme source from the speech engine.
- Browsers only play audio after a click; the agent already starts with a click on the orb, so this is not a problem.

## How it was checked
Chromium test: the four states switch, the mouth follows a test audio (opens and closes, closes again at the end), the
avatar is always shown (no off switch), and the fallback mouth moves when `AudioContext` is missing.
The unit tests in `tests/test_avatar_ui.py` check that the page contains the avatar and its two hooks.
