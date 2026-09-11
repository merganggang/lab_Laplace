"""Calculate an average and save the result as JSON."""

import json
from pathlib import Path
import socket

velocities = [1500, 2000, 2500]
result = {
    "message": "Hello from Python!",
    "container": socket.gethostname(),
    "velocities_m_s": velocities,
    "average_velocity_m_s": sum(velocities) / len(velocities),
}

output = Path("results/hello.json")
output.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(result, indent=2)
output.write_text(text + "\n", encoding="utf-8")
print(text)
print(f"Saved: {output}")
