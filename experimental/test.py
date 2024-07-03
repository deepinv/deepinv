from pathlib import Path

import numpy as np
import torch

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%Y%m%d-%H%M%S")
print("date and time =", dt_string)

a = np.array([1, 2, 3])
b = torch.tensor(a)

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = DATA_DIR / dt_string
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

torch.save(b, SAVE_DIR / "b.pt")
