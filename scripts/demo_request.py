import base64
import io
import json
import random
import sys
import urllib.request

from sysone.envs import tabletop
from sysone.questions import question_to_dict

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1/decide"
scene = tabletop.sample_scene(random.Random(7))
img = tabletop.render(scene)
buf = io.BytesIO()
img.save(buf, format="PNG")
q = tabletop.make_query(scene, with_answers=False)
body = {"state": q.state, "images": [base64.b64encode(buf.getvalue()).decode()],
        "questions": [question_to_dict(x) for x in q.questions]}
req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
print("truth: target", scene["target"], "in_reach", scene["in_reach"], "p_grasp", round(scene["p_grasp"], 2), "clutter", scene["clutter"])
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))
