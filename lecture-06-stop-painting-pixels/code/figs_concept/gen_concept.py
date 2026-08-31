#!/usr/bin/env python3
"""Style-A ("Whiteboard notebook") concept diagrams for Lecture 6.
Reads GEMINI_API_KEY from env. Idempotent: skips figures that already exist.
Run:  python3 gen_concept.py [name ...]
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
    "EXACTLY as specified, perfectly spelled. Spacious, nothing overlapping."
)

SCENES = {
    "loss_two_terms": (
        "Landscape diagram with NO title and NO heading anywhere, explaining the two sides of the I-JEPA loss for ONE target block, laid out as two rows converging on the right. TOP row labeled 'term 1 - the guess': a royal blue rounded box labeled 'predictor' with an arrow to a row of 3 green squares labeled 'predicted tokens'. BOTTOM row labeled 'term 2 - the answer': a royal blue rounded box labeled 'target encoder (teacher)' with an arrow to a row of 3 orange squares labeled 'teacher tokens at the same positions'. RIGHT: the green row sits directly above the orange row; between each vertical PAIR of one green and one orange square is a short crimson double-headed arrow, three small arrows total, with one shared label 'distance, token by token'. Below, a dark ink note: 'average the three distances = loss for this block; repeat for all 4 blocks'. CRITICAL: exactly three pairwise arrows, no arrow between the two blue boxes."
    ),
    "linear_probe": (
        "Landscape diagram explaining a linear probe, two panels. LEFT panel: a royal blue rounded box labeled 'frozen encoder' with a small sketched padlock on it, an arrow in from a small photo labeled 'image', and an arrow out to a row of 4 orange squares labeled 'representation (frozen)'. RIGHT panel: a 2D scatter of small orange dots and small blue dots forming two loose clouds, with ONE straight dark ink line separating the clouds, labeled 'the ONLY thing we train: one straight line'. Below, a note in dark ink: 'if a single line can separate the classes, the encoder already did the understanding'. CRITICAL: the padlock is on the encoder, not on the line."
    ),
    "knn_probe": (
        "Landscape diagram explaining a k-nearest-neighbour probe, one panel. A 2D scatter of many small dots in two soft colors (orange and blue) forming loose clouds; one hollow star labeled 'new image' sits near the orange cloud; thin dark lines connect the star to its 5 closest dots, which are circled; a note beside them reads 'the 5 nearest neighbours vote: 4 orange, 1 blue -> orange'. Below, a dark ink note: 'NOTHING is trained. kNN only works if similar images already sit close together'. CRITICAL: no line separating the clouds anywhere, only the neighbour links."
    ),
    "tsne_concept": (
        "Landscape diagram explaining what a t-SNE map is, two panels with a large arrow between them labeled 't-SNE: keep neighbours together'. LEFT panel titled 'where the encoder puts images': five rows of 6 small orange squares each, stacked, labeled 'representations - 384 numbers each, impossible to draw'. RIGHT panel titled 'the 2D map': a flat scatter of small dots in a few soft colors forming loose clusters, two nearby dots connected by a thin line labeled 'was close in 384-D, stays close in 2-D'. Below, a dark ink note: 'the axes mean nothing; only NEIGHBOURHOODS survive the projection'."
    ),
    "arch_genmask": (
        "Landscape diagram of a generative model that uses block masking, laid out left to right. A sketched photo of a dog with a grid over it where one large rectangular BLOCK of cells is blacked out, labeled 'image with a hidden block'; arrow into a royal blue rounded box labeled 'encoder'; arrow into a royal blue rounded box labeled 'decoder'; arrow out to the same photo with the block filled back in, labeled 'repainted pixels'; a bold crimson double-headed arrow between the repainted photo and a small original photo below it, labeled 'loss: every pixel in the block'. Below, a note in dark ink: 'JEPA masks, but the loss is still paint'."
    ),
    "arch_noguard": (
        "Landscape diagram with NO title and NO heading anywhere - never write the words 'whiteboard notebook' - of I-JEPA with its safety guard removed. Layout: left a sketched masked photo labeled 'context', arrow into a royal blue box labeled 'context encoder', arrow into a royal blue box labeled 'predictor', arrow to a row of 3 green squares labeled 'predicted'; below, the full photo labeled 'target', arrow into a royal blue box labeled 'target encoder', arrow to a row of 3 orange squares labeled 'target'; a crimson double-headed arrow between green and orange rows labeled 'loss'. CRITICAL DIFFERENCE, drawn loudly: a large crimson X scribbled over a small tag reading 'EMA + stop-grad' next to the target encoder, and a crimson arrow flowing INTO the target encoder from the loss labeled 'gradients now flow here too'. Below, a crimson note: 'both encoders can now agree to output a constant'."
    ),

    "arch_jea": (
        "Wide landscape diagram of the Joint-Embedding Architecture, laid out "
        "in three columns left to right. LEFT column: two small photos of the "
        "same dog drawn as sketched picture frames stacked vertically, top one "
        "labeled 'view x' showing the dog facing LEFT in full color, and bottom one labeled 'view x-prime' showing the SAME dog MIRRORED to face RIGHT, cropped closer with a warmer tint - the two views must look clearly DIFFERENT, with a small "
        "note 'two augmentations of one image' beneath them. MIDDLE column: "
        "two royal blue rounded boxes stacked vertically, both labeled "
        "'encoder' with a dashed line between them labeled 'shared weights'. "
        "An arrow goes from 'view x' to the top encoder and from 'view "
        "x-prime' to the bottom encoder. RIGHT column: the top encoder outputs "
        "a row of 4 small orange squares labeled 'z', the bottom encoder "
        "outputs a row of 4 small orange squares labeled 'z-prime'. Between z "
        "and z-prime is a bold crimson red double-headed vertical arrow "
        "labeled 'loss: make them similar'. CRITICAL: there is no decoder and "
        "no pixels on the right side, only the two rows of squares and the "
        "red arrow. Visual story: two views of one image must map to the same "
        "point in representation space."
    ),
    "arch_generative": (
        "Wide landscape diagram of the Generative Architecture, laid out in "
        "four columns left to right. Column 1: a sketched photo of a dog with "
        "a grid over it where several grid cells are blacked out, labeled "
        "'masked image'. Column 2: a royal blue rounded box labeled "
        "'encoder', receiving an arrow from the masked image. Column 3: a "
        "royal blue rounded box labeled 'decoder', receiving an arrow from "
        "the encoder. Column 4: a sketched reconstructed photo of the same "
        "dog labeled 'predicted pixels', and directly below it the original "
        "unmasked photo labeled 'actual pixels'. Between predicted pixels and "
        "actual pixels is a bold crimson red double-headed vertical arrow "
        "labeled 'loss: every pixel'. Visual story: the model must repaint "
        "the missing pixels exactly. CRITICAL: the loss arrow connects the "
        "two photos, not the networks."
    ),
    "arch_jepa": (
        "Wide landscape diagram of the Joint-Embedding Predictive "
        "Architecture, laid out in four columns left to right. Column 1: a "
        "sketched photo of a dog with a grid over it where a rectangular "
        "region is blacked out, labeled 'context x' on top; below it the "
        "complete photo labeled 'target y'. Column 2: two royal blue rounded "
        "boxes: the top labeled 'context encoder' receiving an arrow from "
        "'context x'; the bottom labeled 'target encoder' receiving an arrow "
        "from 'target y'. Column 3: a royal blue rounded box labeled "
        "'predictor' receiving an arrow from the context encoder only. "
        "Column 4: the predictor outputs a row of 4 small green squares "
        "labeled 'predicted representation'; the target encoder outputs a "
        "row of 4 small orange squares labeled 'target representation'. "
        "Between the green row and the orange row is a bold crimson red "
        "double-headed vertical arrow labeled 'loss: in representation "
        "space'. CRITICAL: no decoder anywhere, no predicted pixels anywhere; "
        "the loss compares squares to squares. Visual story: predict the "
        "representation of the hidden part, never repaint pixels."
    ),
    "collapse_intuition": (
        "Landscape diagram explaining representation collapse, two panels "
        "side by side separated by a thin vertical line. LEFT panel titled "
        "'healthy encoder': three different sketched photos (a dog, a car, a "
        "bird) on the left, each with its own arrow into a single royal blue "
        "rounded box labeled 'encoder', and out of the box three SEPARATE "
        "rows of 4 orange squares at three different heights, labeled 'z dog', "
        "'z car', 'z bird', spread apart. RIGHT panel titled 'collapsed "
        "encoder': the same three photos, arrows into an identical royal "
        "blue box labeled 'encoder', but all three arrows out of the box "
        "converge onto ONE single row of 4 gray squares labeled 'same z for "
        "everything'. Below the right panel in crimson red handwriting: "
        "'loss = 0, information = 0'. Visual story: if nothing forbids it, "
        "mapping every image to one constant point is a perfect solution."
    ),
    "ijepa_pipeline": (
        "Very wide landscape diagram of a training pipeline with NO title and "
        "NO caption anywhere — never write the words 'visual story' or any "
        "heading; draw only the diagram. Left to right, TWO horizontal paths "
        "that NEVER cross. TOP path: a 4x4 grid of image patches with most "
        "cells blanked keeping one connected block, labeled 'context block', "
        "flows right into a royal blue rounded box labeled 'context encoder', "
        "which outputs a row of 5 orange squares labeled 'context tokens', "
        "which flows into a royal blue rounded box labeled 'predictor'; a "
        "small tag labeled 'position of target block' also points into the "
        "predictor; the predictor outputs a row of 3 green squares labeled "
        "'predicted' at the TOP RIGHT. BOTTOM path: the full 4x4 image grid "
        "labeled 'full image', flows right into a royal blue rounded box "
        "labeled 'target encoder' with a small gray note 'EMA copy, no "
        "gradient' beneath it and a crimson scissors symbol labeled "
        "'stop-grad' on its output arrow; the target encoder outputs a row "
        "of 3 orange squares labeled 'target' at the BOTTOM RIGHT, directly "
        "below the green 'predicted' row. Between the green row and the "
        "orange row: a bold crimson red double-headed vertical arrow labeled "
        "'loss (repeat for 4 target blocks)'. CRITICAL: the top path touches "
        "only 'predicted'; the bottom path touches only 'target'; the two "
        "paths never cross or exchange arrows."
    ),
    "ema_stopgrad": (
        "Landscape diagram of the EMA teacher trick, two royal blue rounded "
        "boxes side by side with generous space between. Left box labeled "
        "'context encoder (student)', right box labeled 'target encoder "
        "(teacher)'. A thick gray arrow from student to teacher along the "
        "top, labeled 'slow copy of weights: EMA momentum 0.996 to 1.0'. A "
        "crimson red arrow coming up into the student from below labeled "
        "'gradients flow here'. Below the teacher, the same crimson arrow "
        "drawn but crossed out with a big crimson X, labeled 'no gradients, "
        "ever (stop-grad)'. At the bottom center a handwritten note in dark "
        "ink: 'the teacher moves slowly, so the student cannot drag it into "
        "collapse'. Visual story: an asymmetric pair of encoders."
    ),
    "pixel_vs_latent": (
        "Landscape diagram comparing pixel-space and representation-space "
        "prediction, two panels side by side. LEFT panel titled 'predict "
        "pixels': a sketched photo of a dog on grass, with three small "
        "magnifying-glass callouts pointing at the grass, each labeled "
        "'grass blade', and a small callout at the dog labeled 'dog'; below, "
        "a note in dark ink: 'every detail costs the same'. RIGHT panel "
        "titled 'predict representation': the same photo flowing into a row "
        "of 4 orange squares, with three short labels attached to the "
        "squares: 'animal', 'pose', 'position'; below, a note in dark ink: "
        "'only meaning survives'. A thin vertical dividing line between the "
        "panels. Visual story: representation space keeps semantics and "
        "throws away noise."
    ),
    "vit_patches": (
        "Landscape diagram of how a Vision Transformer sees an image, laid "
        "out in three columns left to right. Column 1: a sketched photo "
        "labeled 'image 96 x 96'. Column 2: the same photo cut into a 4x4 "
        "grid with small gaps between cells, labeled 'patches'; a small gray "
        "note '12 x 12 = 144 patches in our runs'. Column 3: a horizontal "
        "row of 8 small orange squares labeled 'tokens', with a small tag "
        "'plus position' attached above; the row flows into a royal blue "
        "rounded box labeled 'transformer'. Visual story: an image becomes a "
        "sequence of tokens, exactly like words in a sentence."
    ),
}


def gen(name):
    path = os.path.join(OUT, f"concept_{name}.png")
    if os.path.exists(path):
        return name, "exists"
    prompt = f"{STYLE}\n\nDIAGRAM: {SCENES[name]}"
    for attempt in range(3):
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
            return name, "no image in response"
        except Exception as e:
            err = str(e)[:120]
    return name, f"failed: {err}"


names = sys.argv[1:] or list(SCENES)
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for name, status in ex.map(gen, names):
        print(f"{name}: {status}")
