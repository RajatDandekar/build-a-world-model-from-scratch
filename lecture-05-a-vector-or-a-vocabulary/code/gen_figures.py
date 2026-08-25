#!/usr/bin/env python3
"""Style A ("whiteboard notebook") figures for Lecture 4 — the RSSM on SO-101.

Full-slide diagrams: one idea per figure, generous space, exact labels.
Run on PLAIN python3 (google-genai lives there): python3 gen_figures.py [name ...]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

# Bring your own key:  export GEMINI_API_KEY=...
if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("set GEMINI_API_KEY in your environment first")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "img")

STYLE = (
    "STYLE 'WHITEBOARD NOTEBOOK': a technical diagram drawn in a tidy hand-sketched "
    "Excalidraw style on a very light grid-paper background. Thin, slightly wobbly ink "
    "strokes; handwritten-looking but perfectly legible casual lettering. Colors: dark "
    "ink for text, royal blue for network shapes, orange for data blocks, crimson red "
    "for emphasis and loss, teal for memory contents. Vectors are drawn as rows of "
    "small hand-drawn squares. Charming but precise, like the best lecture notes you "
    "have ever seen. All labels EXACTLY as specified, perfectly spelled. Spacious, "
    "nothing overlapping, no crossing arrows.")

SCENES = {

"fig_running_memory": (
 "A diagram titled 'A RUNNING SUMMARY' showing a newspaper article being read left to right. "
 "On the LEFT, a hand-drawn newspaper page with a headline bar and several lines of wavy "
 "'text' squiggles, labeled 'a news article'. From it, five orange squares run left to right "
 "across the middle of the figure, each containing one legible word: 'the', 'minister', "
 "'from', 'harlem', 'resigned'. "
 "ABOVE the row of words sits ONE teal rectangle labeled 'the summary (48 numbers)'. From "
 "each word square a short arrow points UP into that same teal rectangle. A small circular "
 "arrow on the rectangle is labeled 'rewritten at every word'. "
 "On the far RIGHT, a crimson red question mark and the handwritten note 'what is left of "
 "the first word by now?'. "
 "Bottom caption in dark ink: 'one box, one size, no matter how long the article'."),

"fig_the_task": (
 "A diagram titled 'THE TASK WE WILL MEASURE' laid out as one long horizontal strip of text "
 "boxes, left to right. "
 "FIRST box, teal outlined, containing the words 'minister IBARRA was named'. Under it the "
 "label 'the answer, stated once'. "
 "MIDDLE: a long orange box containing faded repeated filler words 'the committee met again "
 "the following week the report was circulated' with an ellipsis. Under it the label 'N words "
 "of filler'. Above it a double-headed arrow labeled 'we vary N: 4, 12, 24, 48, 96, 160'. "
 "LAST box, crimson red outlined, containing 'who resigned ?'. Under it the label 'the "
 "question'. "
 "Below everything, a wide dark ink caption: 'the answer is always in the passage. the ONLY "
 "thing we change is how far back it sits.'"),

"fig_accuracy": (
 "A diagram titled 'HOW WE SCORE IT' in two stacked halves. "
 "TOP half: a small royal blue rounded box labeled 'the model' with an arrow coming in from "
 "the left labeled 'one passage' and an arrow going out to the right into a vertical list of "
 "EIGHT names, each in a small box: 'patel', 'okafor', 'novak', 'silva', 'haddad', 'ibarra', "
 "'kaur', 'moreau'. The box for 'ibarra' is highlighted in teal with a tick mark beside it, "
 "and a handwritten label 'the model picks ONE'. "
 "BOTTOM half: a row of 10 small squares, 7 of them with teal ticks and 3 with crimson "
 "crosses, and beneath it the handwritten formula-free label 'accuracy = how often the pick "
 "is right, over 400 fresh passages'. "
 "To the right, a small dark ink note: 'guessing at random would be 1 in 8 = 0.125'."),

"fig_attention_intro": (
 "A diagram titled 'ATTENTION, IN ONE PICTURE' with a single sentence laid out as six orange "
 "squares in a row along the BOTTOM, each with one legible word: 'minister', 'ibarra', 'was', "
 "'named', 'who', 'resigned'. "
 "ABOVE the last word 'resigned' sits a royal blue circle labeled 'the question'. From that "
 "circle, six thin arrows fan out and point DOWN to each of the six words. Five arrows are "
 "thin and pale grey. The ONE arrow pointing to 'ibarra' is thick and crimson red. "
 "Beside the thick red arrow, a handwritten label reads '0.25 of the attention'. Beside the "
 "pale arrows, a small label reads 'almost nothing'. "
 "On the right, three short stacked handwritten lines: 'QUERY: what am I looking for?', "
 "'KEY: what do I have?', 'VALUE: here is what you get'. "
 "Bottom caption in dark ink: 'nothing is summarised. it looks, and it chooses.'"),

"fig_parallel_training": (
 "A comparison diagram titled 'WHY TRANSFORMERS TRAIN FAST' split by a horizontal line. "
 "TOP half headed 'RNN: one at a time'. Five orange word squares 'w1' to 'w5' in a row, "
 "connected LEFT TO RIGHT by thick arrows in a single chain. Above them, five small clocks "
 "numbered 'step 1', 'step 2', 'step 3', 'step 4', 'step 5' arranged left to right. A "
 "crimson red handwritten note: 'w5 cannot start until w4 has finished'. "
 "BOTTOM half headed 'TRANSFORMER: all at once'. The same five orange squares 'w1' to 'w5' in "
 "a row, but with NO chain between them. Instead one wide royal blue bar sits above all five "
 "at once, labeled 'one forward pass', with five short arrows going down to all five squares "
 "simultaneously. Above it a SINGLE clock labeled 'step 1'. A teal handwritten note: 'five "
 "predictions, one pass — because the mask already hides each word's future'."),

"fig_vq_losses": (
 "A diagram titled 'THE TWO FORCES ON A TOKENIZER' showing a horizontal pipeline. "
 "LEFT: a small dark game frame labeled 'the frame'. An arrow to a royal blue box 'encoder'. "
 "An arrow to a single teal square labeled 'what the encoder produced'. "
 "MIDDLE: an orange rounded box labeled 'the codebook' containing three small squares "
 "labeled 'word 41', 'word 99', 'word 260'. A dashed arrow from the teal square to 'word 99' "
 "labeled 'snap to the nearest'. Between the teal square and 'word 99' there is a small gap, "
 "and a double-headed crimson arrow across that gap labeled 'LOSS 2: pull these two "
 "together'. "
 "RIGHT: an arrow from the codebook to a royal blue box 'decoder', then to a second game "
 "frame labeled 'the frame, rebuilt'. A double-headed crimson arrow arcs from 'the frame' on "
 "the far left to 'the frame, rebuilt' on the far right, labeled 'LOSS 1: make these two "
 "match'. "
 "Bottom caption in dark ink: 'loss 1 trains the encoder and decoder. loss 2 keeps the "
 "dictionary near what the encoder actually says.'"),

"fig_iris_overview": (
 "A large system diagram titled 'IRIS, END TO END' with three numbered stages laid out left "
 "to right, each in its own dashed outlined region. "
 "STAGE 1, headed 'the tokenizer (VQ-VAE)': a small game frame, an arrow into a royal blue "
 "box 'encoder / decoder', an arrow out to a row of four small orange squares containing the "
 "numbers '178', '277', '99', '500', labeled '16 words per frame'. Under the region: 'trained "
 "FIRST, then frozen'. "
 "STAGE 2, headed 'the dynamics model (transformer)': a long row of orange squares with one "
 "blue square after every group, labeled 'frame words + action, frame words + action', "
 "feeding UP into a wide royal blue bar labeled 'causal transformer'. An arrow comes out of "
 "the right end into an orange bell-curve labeled 'which word comes next?'. Under the region: "
 "'trained SECOND, on tokens only'. "
 "STAGE 3, headed 'imagination': a small loop of three game frames connected by curved arrows "
 "labeled 'action in, frame out, repeat'. Under the region: 'no game engine anywhere'. "
 "Bottom caption in dark ink: 'IN: past frames + past actions.   OUT: the next frame, one "
 "word at a time.'"),

"fig_transformer_overview": (
 "A vertical architecture diagram titled 'THE WHOLE STACK, AT A GLANCE', read BOTTOM to TOP, "
 "spacious and clean. "
 "At the BOTTOM, a row of orange squares containing '178', '277', '99', and one blue square "
 "containing 'a', labeled 'the tokens'. "
 "Above it a narrow box 'look up the embedding table' and then a small crimson circle with a "
 "plus sign labeled 'add position'. "
 "Above that, THREE identical stacked royal blue rounded boxes, each labeled 'ATTENTION + "
 "MLP', with a curly brace to the right of them labeled 'x 6 layers'. Small dots between the "
 "second and third box to suggest repetition. "
 "Above them a narrow box 'LayerNorm', then a narrow box 'one matrix -> 512 scores'. "
 "At the TOP an orange bell curve labeled 'a probability for each of the 512 words'. "
 "On the LEFT, running the whole height of the diagram, a thick vertical teal arrow labeled "
 "'the residual stream: 256 numbers, all the way up'. "
 "Bottom caption in dark ink: 'everything between the tokens and the scores is the same block, "
 "repeated.'"),

"fig_train_vs_dream": (
 "A comparison diagram titled 'TRAINING IS PARALLEL. DREAMING IS NOT.' split by a horizontal "
 "line. "
 "TOP half headed 'TRAINING: one pass, 136 predictions'. A long row of small orange squares "
 "labeled underneath 'all 136 positions, already known'. Above the row, one wide royal blue "
 "bar labeled 'ONE forward pass'. From the bar, many short arrows point down to every square "
 "at once. A teal handwritten note: 'we already have the true next token everywhere, so every "
 "position can be scored at the same time'. "
 "BOTTOM half headed 'DREAMING: 16 passes, one frame'. Four small orange squares in a row "
 "with a crimson question mark box at the end. Above them, FOUR separate small royal blue "
 "bars numbered 'pass 1', 'pass 2', 'pass 3', 'pass 4', arranged in a rising staircase, each "
 "with an arrow down to the next square in turn. A crimson handwritten note: 'each word must "
 "be written down before the next one can be asked for'."),

"fig_context_probe": (
 "A diagram titled 'HOW WE MEASURED HOW FAR BACK IT LOOKS' showing four stacked rows, each a "
 "horizontal strip of eight small frame boxes labeled 'f1' to 'f8'. "
 "ROW 1: all eight boxes are orange and visible. Label to the right: 'show all 8 -> error "
 "0.019'. "
 "ROW 2: the first five boxes are greyed out and crossed with a light X, the last three are "
 "orange. Label: 'hide all but 3 -> error 0.019'. "
 "ROW 3: the first six boxes greyed and crossed, the last two orange. Label: 'hide all but 2 "
 "-> error 0.099'. "
 "ROW 4: the first seven boxes greyed and crossed, only the last orange. Label to the right "
 "in the same dark ink as the other rows: 'hide all but 1 -> error 1.57', but with the "
 "number '1.57' alone circled in crimson red for emphasis. "
 "A single vertical crimson bracket runs down the RIGHT-HAND edge of the last column, "
 "spanning all four rows, with ONE label written beside it exactly once: 'we always score "
 "the SAME thing: the newest frame'. Do not repeat this label on individual rows. "
 "Bottom caption in dark ink: 'blank out the oldest frames one at a time and watch what "
 "breaks. the RSSM has no such knob.'"),


"fig_block_anatomy": (
 "A vertical flow diagram titled 'ONE BLOCK OF THE STACK', drawn bottom to top. "
 "At the BOTTOM a horizontal row of small teal squares labeled 'the vector at one position "
 "(256 numbers)'. A thick vertical arrow runs all the way from bottom to top on the LEFT "
 "side of the figure, labeled 'the residual stream'. "
 "Going up the middle: first a small rounded box labeled 'LayerNorm', then a wide royal "
 "blue rounded box labeled 'ATTENTION (8 heads)' with a small note beside it in dark ink "
 "reading 'the only place positions talk to each other'. From the attention box an arrow "
 "goes right and then into a crimson red circle containing a plus sign, which also "
 "receives the thick residual arrow from below. Label under the plus circle: 'add, do not "
 "replace'. "
 "Continuing up: another small rounded box labeled 'LayerNorm', then a wide orange rounded "
 "box labeled 'MLP  256 -> 1024 -> 256' with a note beside it reading 'each position on "
 "its own'. Then a second crimson red plus circle, again fed by the residual arrow. "
 "At the very TOP a row of teal squares labeled 'out, same 256 numbers'. "
 "On the far right, outside everything, a tall curly brace spanning the whole diagram with "
 "the handwritten label 'x 6 layers'."),

"fig_stream_river": (
 "A diagram titled 'THE RESIDUAL STREAM' showing one thick horizontal band running left to "
 "right across the whole figure, drawn as a wide teal channel. "
 "At the FAR LEFT the band starts narrow, labeled below 'word 99, looked up in the "
 "embedding table'. At the FAR RIGHT the band is much thicker, labeled below 'a summary of "
 "the whole 136-token window'. "
 "Along the band, six evenly spaced royal blue rounded boxes sit ABOVE it, labeled 'layer "
 "1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'layer 6'. From each box a short arrow "
 "points DOWN into the band, and each arrow is labeled 'adds'. "
 "Handwritten note above the left end: 'nothing is ever overwritten'. Handwritten note "
 "below the right end in crimson red: 'similarity to the starting word: 1.00 -> 0.20'. "
 "Small dark ink caption centered underneath the whole band: 'the same vector, all the "
 "way through'."),


"fig_rnn_vs_attn": (
 "A comparison diagram titled 'TWO WAYS TO READ A SEQUENCE' with a horizontal dividing "
 "line across the middle. "
 "TOP half headed 'the recurrent way': five small orange squares in a row along the "
 "bottom labeled 'w1', 'w2', 'w3', 'w4', 'w5'. Above them a single teal framed rectangle "
 "labeled 'the summary'. Each word square has a short arrow going UP into the SAME teal "
 "rectangle, and the rectangle has a small circular arrow on itself labeled 'rewritten "
 "each step'. A handwritten note reads: 'everything must fit in here'. "
 "BOTTOM half headed 'the attention way': the same five orange squares labeled 'w1' to "
 "'w5', all kept. Above the last square is a small royal blue circle labeled 'now'. From "
 "'now', five separate thin arrows fan DOWN and BACK to each of the five word squares, "
 "with three of the arrows drawn thicker than the others. A handwritten note reads: "
 "'nothing is thrown away - it looks back'. "
 "CRITICAL: in the top half every arrow points into one rectangle; in the bottom half "
 "the arrows fan out from one point to many. Use exactly these labels."),

"fig_two_latents": (
 "A comparison diagram titled 'WHAT CAN A LATENT BE?' with a vertical dividing line down "
 "the middle. "
 "LEFT half headed 'CONTINUOUS' with subtitle 'a point on a map': draw a loose scattered "
 "cloud of small dots, with two dots highlighted in crimson red and a smooth dashed line "
 "drawn between them with three small circles sitting ON the line between them. Label "
 "under it: 'halfway between two points is another point'. "
 "RIGHT half headed 'DISCRETE' with subtitle 'a word from a list': draw a neat vertical "
 "list of six small orange rounded boxes labeled 'word 1', 'word 2', 'word 3', 'word 4', "
 "'word 5', 'word 6'. Between 'word 2' and 'word 3' draw a crimson red question mark and "
 "a small crimson red cross. Label under it: 'there is nothing in between'. "
 "CRITICAL: the left half shows a continuous cloud with an interpolation path; the right "
 "half shows a discrete enumerated list with a gap. No arrows cross the middle."),

"fig_average_vs_choose": (
 "A diagram titled 'THE MATCH WAS CANCELLED BECAUSE OF THE ___' laid out as two columns "
 "with a vertical dividing line. "
 "LEFT column headed 'one Gaussian must answer with a point': three small teal dots "
 "spread apart, labeled 'rain', 'snow' and 'strike', and a single large crimson red X "
 "drawn at the centre between them labeled 'the average'. Below, a small blurred smudge "
 "shape and the handwritten line 'decode that and you get a word that does not exist'. "
 "RIGHT column headed 'a vocabulary can answer with a choice': three horizontal orange "
 "bars of different lengths, labeled at their left ends 'rain', 'snow', 'strike', with "
 "the numbers '0.5', '0.3' and '0.2' written at the right end of each bar. Below, the "
 "handwritten line 'every answer it names is a real word'. "
 "CRITICAL: exactly these labels. The left side has one X between three dots; the right "
 "side has three labelled bars."),

"fig_vq_how": (
 "A left-to-right pipeline diagram titled 'HOW A PICTURE BECOMES WORDS' with four stages "
 "and generous space between them. "
 "STAGE 1: a small dark game screenshot labeled 'the frame'. "
 "STAGE 2: an arrow into a royal blue rounded box labeled 'encoder', with an arrow out to "
 "a small 4 by 4 grid of empty squares labeled 'a 4 x 4 grid of vectors'. "
 "STAGE 3: BELOW the grid, a tall orange rounded rectangle labeled 'the codebook' "
 "containing six small numbered squares labeled '1', '2', '3', '...', '511', '512'. From "
 "the 4 by 4 grid, one dashed arrow points down to the codebook, labeled 'snap each "
 "square to its nearest entry'. To the right of the codebook, a 4 by 4 grid of small "
 "squares each containing a number, labeled '16 whole numbers'. "
 "STAGE 4: an arrow from that grid of numbers into a royal blue rounded box labeled "
 "'decoder', with an arrow out to a small dark game screenshot labeled 'the frame, "
 "rebuilt'. "
 "A handwritten note along the bottom reads: 'the frame is now a short sentence'."),

"fig_iris_loop": (
 "A diagram titled 'THE WORLD MODEL AS A LANGUAGE MODEL' showing a horizontal sequence "
 "running left to right along the middle of the page. "
 "The sequence is a row of small orange squares grouped into three labelled clusters. "
 "Cluster one is labeled below as 'frame t-2' and contains four small orange squares "
 "followed by one blue square labeled 'a'. Cluster two is labeled 'frame t-1' with the "
 "same pattern. Cluster three is labeled 'frame t' with the same pattern. "
 "ABOVE the whole sequence, one wide royal blue rounded box labeled 'transformer', with "
 "several thin arrows going down from the box to the squares below it. "
 "To the RIGHT of the sequence, an arrow leads to a small orange bell curve labeled "
 "'which word comes next?', and from the curve a dashed arrow to a small group of four "
 "orange squares labeled 'frame t+1'. "
 "A handwritten note at the bottom reads: 'exactly how a language model writes the next "
 "word'. "
 "CRITICAL: use exactly these labels, and keep the sequence in one straight horizontal "
 "line."),

"fig_family_tree": (
 "A clean two-by-two grid diagram titled 'FOUR CORNERS, ALL OCCUPIED', drawn as a large "
 "square divided into four equal quadrants by one horizontal and one vertical line. "
 "Label along the LEFT edge, rotated, the two rows: top row 'RECURRENT' and bottom row "
 "'ATTENTION'. Label along the TOP edge the two columns: left column 'CONTINUOUS latent' "
 "and right column 'DISCRETE latent'. "
 "Inside the top-left quadrant write 'RSSM' in large letters and beneath it in smaller "
 "letters 'PlaNet 2019'. "
 "Inside the top-right quadrant write 'DreamerV2' and beneath it '2020'. "
 "Inside the bottom-right quadrant write 'IRIS' and beneath it '2022'. "
 "Inside the bottom-left quadrant write 'Dreamer 4' and beneath it '2025'. "
 "CRITICAL: exactly four quadrants, exactly these eight labels, nothing else inside the "
 "square. Keep the grid perfectly square and the labels centred in their quadrants."),

"fig_cost_per_step": (
 "A diagram titled 'WHAT EACH STEP COSTS' with two panels side by side, separated by a "
 "vertical line. "
 "LEFT panel headed 'the recurrent state': four identical small teal squares in a "
 "horizontal row connected by arrows, each square exactly the same size, with a small "
 "clock symbol of the SAME size drawn under each one. Caption below: 'every step costs "
 "the same, forever'. "
 "RIGHT panel headed 'the attention window': four groups of orange squares in a "
 "horizontal row where each group is BIGGER than the one before it - the first group has "
 "one square, the second two, the third three, the fourth four - and under each group a "
 "clock symbol that grows correspondingly larger. Caption below: 'every step costs more "
 "than the last'. "
 "CRITICAL: on the left everything is the same size; on the right things grow to the "
 "right. Use exactly these captions."),
}


def generate(name):
    prompt = STYLE + "\n\nDIAGRAM: " + SCENES[name]
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9")))
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    path = os.path.join(OUT, name + ".png")
                    with open(path, "wb") as fh:
                        fh.write(part.inline_data.data)
                    return f"{name}: {len(part.inline_data.data)//1024} KB"
        except Exception as exc:                                  # noqa: BLE001
            if attempt == 2:
                return f"{name}: FAILED {exc}"
    return f"{name}: no image returned"


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENES)
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in pool.map(generate, names):
            print(line)
