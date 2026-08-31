#!/usr/bin/env python3
"""Style-A concept diagrams for the Motivation deck (lecture 6a).
Same style/protocol as gen_concept.py. Run: python3 gen_motivation.py [names]
"""
import os
import sys
import concurrent.futures as cf

from google import genai
from google.genai import types

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("set GEMINI_API_KEY first")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
OUT = os.path.dirname(os.path.abspath(__file__))

STYLE = (
    "STYLE 'WHITEBOARD NOTEBOOK': a technical diagram drawn in a tidy "
    "hand-sketched Excalidraw style on a very light grid-paper background. "
    "Thin, slightly wobbly ink strokes; handwritten-looking but perfectly "
    "legible casual lettering. Colors: dark ink for text, royal blue for "
    "network shapes, orange for data blocks, crimson red for emphasis and "
    "loss. Vectors are drawn as rows of small hand-drawn squares. Charming "
    "but precise, like the best lecture notes you've ever seen. All labels "
    "EXACTLY as specified, perfectly spelled. Spacious, nothing overlapping. "
    "NO title and NO caption anywhere; draw only the diagram."
)

SCENES = {
    "brain_simulator": (
        "Landscape diagram of mental simulation while driving near a cliff. "
        "LEFT: a simple sketched side profile of a human head; inside the "
        "head a royal blue rounded box labeled 'world model'. An arrow from "
        "the eyes labeled 'perception' enters the box. RIGHT of the head: "
        "a filmstrip of three small sketched frames stacked left to right "
        "showing a car approaching a cliff edge, then the car at the very "
        "edge tipping, then a crimson X over the falling car; the filmstrip "
        "is labeled 'imagined rollout — never executed'. An arrow from the "
        "world model box to the filmstrip labeled 'predict'. Below the "
        "filmstrip a note in dark ink: 'the accident is simulated, not "
        "experienced'. CRITICAL: the imagined frames are drawn inside a "
        "thought-bubble outline to show they are internal."
    ),
    "modalities_one_arch": (
        "Landscape diagram showing one architecture handling three "
        "modalities, laid out in three columns. LEFT column, three sketched "
        "input pairs stacked vertically with labels: top 'image: two crops "
        "of one photo' (two small photo frames), middle 'video: frame t and "
        "frame t+1' (two filmstrip frames), bottom 'audio: clip and its "
        "continuation' (two small waveform snippets). MIDDLE column: ONE "
        "royal blue rounded box labeled 'the same JEPA' with three arrows "
        "entering it, one from each pair. RIGHT column: a row of 4 orange "
        "squares labeled 'representation of x' above a row of 4 orange "
        "squares labeled 'representation of y', with a bold crimson "
        "double-headed arrow between them labeled 'predict one from the "
        "other'. Visual story: the loss never mentions pixels, samples or "
        "waveforms — so the recipe is modality-agnostic."
    ),
    "camera_latent_concept": (
        "Landscape diagram of the camera-shift latent variable. LEFT: two "
        "sketched frames of the same simple scene (a tree and a house), the "
        "second frame shifted a little down and right; labeled 'frame t' "
        "and 'frame t+1'. Between them a note: 'everything moved together'. "
        "MIDDLE: a royal blue rounded box labeled 'predictor' receiving an "
        "arrow from 'frame t' representation (a row of 3 orange squares) "
        "AND a small side arrow from a tag labeled 'z = camera motion "
        "(dx, dy)'. RIGHT: a row of 3 green squares labeled 'predicted "
        "representation of frame t+1'. Below, a note in dark ink: 'z "
        "carries exactly what the past cannot know'. CRITICAL: the z tag is "
        "drawn small, with a note 'only 2 numbers' in gray."
    ),
    "energy_valleys": (
        "Landscape diagram comparing two energy landscapes as simple 1D "
        "curves, two panels side by side. LEFT panel titled in dark ink "
        "'useful model': a hand-drawn curve with two deep valleys, and two "
        "small orange dots sitting at the bottoms of the two valleys "
        "labeled 'plausible future A' and 'plausible future B'; high curve "
        "regions labeled 'implausible futures' in gray. RIGHT panel titled "
        "'collapsed model': an almost flat horizontal line with the note in "
        "crimson 'every future has low energy — the model knows nothing'. "
        "The x axis of both panels is labeled 'all possible futures y' and "
        "the y axis 'energy E(x, y)'."
    ),
    "capacity_pipe": (
        "Landscape diagram about limiting the information in the latent "
        "variable, two panels side by side. LEFT panel titled 'z too big': "
        "a WIDE gray pipe labeled 'z: unlimited bits' feeding into a royal "
        "blue box labeled 'predictor'; below it a crimson note 'z can "
        "smuggle in the whole answer — every y gets low energy'. RIGHT "
        "panel titled 'z limited': a NARROW gray pipe labeled 'z: a few "
        "bits' feeding an identical predictor box; below it a dark ink "
        "note 'z can only pick among a few futures — energy stays low only "
        "near real data'. CRITICAL: the two pipes must differ dramatically "
        "in width."
    ),
    "baby_physics": (
        "Landscape diagram of what infants learn without labels, drawn as "
        "a simple horizontal timeline from left to right labeled 'age in "
        "months'. Four ticks at 2, 4, 9 and 12 months. Above each tick a "
        "tiny sketched vignette with a label: at 2 months a face and the "
        "label 'faces, object unity'; at 4 months a ball rolling behind a "
        "screen labeled 'object permanence'; at 9 months a cup falling "
        "labeled 'gravity, support'; at 12 months a stacked tower labeled "
        "'stability, containment'. Below the timeline a dark ink note: "
        "'no labels, no rewards — just watching'."
    ),
}


def gen(name):
    path = os.path.join(OUT, f"concept_{name}.png")
    if os.path.exists(path):
        return name, "exists"
    prompt = f"{STYLE}\n\nDIAGRAM: {SCENES[name]}"
    err = "?"
    for _ in range(3):
        try:
            r = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9"),
                ),
            )
            for part in r.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    open(path, "wb").write(part.inline_data.data)
                    return name, "ok"
            return name, "no image"
        except Exception as e:
            err = str(e)[:100]
    return name, f"failed: {err}"


names = sys.argv[1:] or list(SCENES)
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    for name, status in ex.map(gen, names):
        print(f"{name}: {status}")
