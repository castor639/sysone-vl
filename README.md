# imagen

Pictures made by a model that can only answer questions.

<p align="center">
  <img src="docs/imagen/crescent_moon.png" width="150" alt="crescent moon">
  <img src="docs/imagen/full_moon.png" width="150" alt="full moon">
  <img src="docs/imagen/sun.png" width="150" alt="sun">
  <img src="docs/imagen/rose.png" width="150" alt="rose">
  <img src="docs/imagen/beach.png" width="150" alt="beach">
</p>
<p align="center"><i>"crescent moon", "full moon", "sun", "rose", "beach". 64 by 64 cells, unedited.</i></p>

See it live at [sysone.warpfield.me](https://sysone.warpfield.me).

## What it does

You type a prompt like "a sun in the sky over grass." imagen never asks a model to draw it. Instead it asks [Jev](https://typesafe.ai), TypeSafe's hosted decision model, a long list of typed questions. Jev writes no text. It only returns probabilities.

There are three kinds of question:

- **Choice:** pick one option from a list. Jev gives a probability for each option.
- **Score:** rate something on a scale. Jev gives a probability for each level.
- **Noul:** a yes or no question. Jev gives one probability.

Code does the geometry. Jev makes every decision about what goes in the picture.

## How it draws

1. **Layout.** Jev picks the main subject from the words in your prompt. It also picks a shape, a size, a spot on the canvas and the colors. It decides if there is a horizon and how high it sits. Code turns those answers into a plan.
2. **Pixels, coarse to fine.** Code builds the picture at a quarter size, then half size, then full size. Each cell is one Choice over a 16 color palette. The question tells Jev where the cell sits, which part of the plan it falls in and what its neighbors look like. Cells deep inside a solid area copy their parent, so Jev only gets asked about the cells that matter.
3. **Render.** Each cell is a blend of the palette, weighted by Jev's probabilities. That blend gives the pictures their soft edges and shading. Pass `--render hard` if you want flat colors instead.

<p align="center">
  <img src="site/img/full_moon_plan.svg" width="260" alt="layout plan for full moon">
  <img src="docs/imagen/full_moon.png" width="260" alt="finished full moon">
</p>
<p align="center"><i>Left: the layout Jev chose for "full moon". Right: the finished picture.</i></p>

## Numbers

These are the five pictures above, one run each from the chat page. The raw run reports are in [`docs/imagen/reports/`](docs/imagen/reports).

| Prompt | Time | Jev calls | Cells asked | Input tokens |
|---|---|---|---|---|
| crescent moon | 4.9 s | 16 | 1,128 | 0.38M |
| full moon | 5.2 s | 21 | 1,668 | 0.57M |
| sun | 4.9 s | 20 | 1,578 | 0.53M |
| rose | 3.6 s | 12 | 894 | 0.30M |
| beach | 3.6 s | 15 | 972 | 0.34M |

## Run it

You need Python 3.10 or newer and a TypeSafe API key. From the repo folder:

```bash
pip install httpx pillow fastapi uvicorn
export TYPESAFE_API_KEY=your-key-here
python -m imagen "a sun in the sky over grass" --grid 64 --out out/sun
```

The finished picture lands in `out/sun/best.png`. You also get one PNG per round and a `report.json` with the plan and the token count.

Options:

- `--grid` sets the canvas size in cells, from 16 to 128. The default is 32.
- `--render hard` uses flat palette colors instead of the soft blend.
- `--rounds` sets how many rounds each level runs. The default is 2.
- `--seed` fixes the random choices so you can repeat a run.

There is also a chat page that shows each round as it comes in:

```bash
python -m imagen.app
```

Then open http://127.0.0.1:7860.

The imagen tests mock the API, so they need no key:

```bash
pip install pytest
python -m pytest tests/test_imagen.py
```

## The experiment behind it: sysone-vl

Before imagen, we built our own typed decision model for robots. It looks at a camera image of a tabletop and answers four questions in one pass, with no text generated. It runs on Qwen2.5-VL-3B with small answer heads on top and a LoRA adapter on the language model.

<p align="center">
  <img src="site/img/scene.svg" width="340" alt="tabletop scene with a gripper, a grey block on a green triangle, and three other objects">
</p>

Here is one scene from the simulator. The task is "pick up the object closest to the gripper." The four questions and the true answers are:

| Question | True answer |
|---|---|
| Which object should the gripper pick? | green triangle |
| Is the target inside the gripper's reach? | yes |
| Will a grasp on it work right now? | one time in four |
| How cluttered is the table? | heavy |

The grey block sits on top of the triangle. That is why a grasp only works one time in four.

The hard part is that last kind of answer. Plain supervised training only ever sees "worked" or "failed," so the model learns to guess close to 0 or 1. We added a second stage called RLCD (reinforcement learning for calibrated decisions). The model tries grasps in the simulator, sees what happens and gets scored on how good its probability was.

## sysone-vl results

All numbers come from `runs/*/eval.json` and `runs/baseline_generate.json`. Each test set is 500 new scenes, except the baseline, which used 100.

| | Supervised only | After RLCD |
|---|---|---|
| Picks the right object | 98.8% | 99.2% |
| Gets reach right | 88.2% | 92.4% |
| Grasp calibration error | 0.21 | 0.06 |
| Grasp odds, average miss vs true odds | 0.24 | 0.13 |
| Time for one four-question query | 108 ms | 110 ms |

Calibration error measures how far the model's stated odds are from how often grasps really work. Lower is better. We also tried the usual quick fix, temperature scaling, on the supervised model. Grasp calibration error stayed at 0.21. RLCD is what moved it.

For comparison, the untrained base model, asked to reply with a letter, picked the right object 69% of the time.

<p align="center">
  <img src="docs/reliability.png" width="720" alt="calibration plots before and after RLCD">
</p>

## Train sysone-vl yourself

You need an NVIDIA GPU. Our run used one L40S, and each training stage took about 11 minutes there. The trained weights and the generated data are not in this repo, so you make them yourself:

```bash
pip install -e .
bash scripts/run_pipeline.sh
```

That script builds the tabletop dataset, runs supervised training, fits temperatures, runs RLCD, evaluates each step and runs the base model baseline. You can change the size with `TRAIN`, `EPOCHS` and `RLCD_STEPS`, for example `TRAIN=1000 RLCD_STEPS=100 bash scripts/run_pipeline.sh`.

To serve the trained model and send it the scene above:

```bash
python -m sysone.serve --ckpt runs/rlcd --port 8000
python scripts/demo_request.py http://localhost:8000/v1/decide
```

The simulator test runs on any machine:

```bash
python -m pytest tests/test_env.py
```

## Limits

- imagen needs a paid TypeSafe API key. Jev is a hosted model and is not part of this repo.
- imagen makes small pixel art with a sensible layout. It does not make photographs. The numbers above come from one run of each of five prompts.
- sysone-vl was trained and tested only on a simple simulated tabletop with flat shapes. It has never seen a real robot or a real camera.
- The sysone-vl weights and data are not included. You need a GPU to train them, and `tests/test_tokenizer.py` downloads the Qwen model files the first time it runs.

## Repo layout

- `imagen/` the image generator: Jev client, layout and pixel questions, command line and chat page
- `sysone/` the vision decision model: tabletop simulator, model, training and evaluation
- `scripts/` the training pipeline, the demo request and the figure script
- `runs/` configs, training logs and eval results from our run
- `docs/` figures
- `site/` the project website

## Credits

Jev and its Choice, Score and Noul types come from TypeSafe. imagen calls their hosted API. sysone-vl is our own build of the same idea on an open vision model, based on TypeSafe's public description of Jev.

## License

MIT. See [LICENSE](LICENSE).

---

A Warpfield project
