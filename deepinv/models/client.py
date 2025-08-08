import torch
import requests
import io
import base64
import json
from warnings import warn

from deepinv.models import Reconstructor, Denoiser


class Client(Reconstructor, Denoiser):
    r"""
    DeepInverse model API Client.

    Interact with model APIs directly from DeepInverse.

    During forward pass, passes input tensor serialized as base64 to API, along with any optional params,
    which must either be plain text, numbers, or serializable, depending on the API input requirements,
    such as `physics` string, `config`, `sigma`, `mask` etc.

    **API DOCS**

    All APIs wishing to be used with Client must follow:

    * Since we cannot pass objects via the API, physics are passed as strings with optional parameters and must be rebuilt in the API.
    * The API must accept the following input body:

    ```python
    {
        "input": {
            "file": <b64 serialized file>,
            "param1": "such as a config str",
            "param2": <or a b64 serialized param>,
        }
    }
    ```

    * The API must return the following output response:

    ```python
    {
        "output": {
            "file": "<b64 serialized file>",
            "other_outputs": "such as inference time",
        }
    }
    ```

    :param str api_key: API key.
    :param str endpoint: endpoint URL.
    """

    def __init__(self, endpoint: str, api_key: str = ""):
        super().__init__(device=None)
        self.api_key = api_key
        self.endpoint = endpoint
        self.training = False

    def serialize(self, tensor: torch.Tensor) -> str:
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def deserialize(self, data: str) -> torch.Tensor:
        buffer = io.BytesIO(base64.b64decode(data))
        return torch.load(buffer, map_location="cpu")

    def check_value(self, v):
        ALLOWED = (int, float, str, bool, type(None))
        if isinstance(v, torch.Tensor):
            return self.serialize(v)
        if isinstance(v, ALLOWED):
            return v
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, ALLOWED) for x in v):
                return v
            raise TypeError("Lists/tuples may only contain primitive types")
        raise TypeError(f"Unsupported kwarg value type: {type(v).__name__}")

    def forward(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.training:
            raise RuntimeError("Model client can only be used in evaluation mode.")

        safe_kwargs = {
            k: self.check_value(v) for k, v in kwargs.items() if isinstance(k, str)
        }
        if len(safe_kwargs) != len(kwargs):
            raise TypeError("All kwarg keys must be strings")

        payload = {"input": {"file": self.serialize(y), **safe_kwargs}}

        headers = {"Content-Type": "application/json"}
        if self.api_key != "":
            headers |= {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(
            self.endpoint, headers=headers, data=json.dumps(payload)
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"API call failed: {response.status_code} - {response.text}"
            )

        result = response.json()

        if "output" not in result:
            raise ValueError("Response missing 'output' field")

        if "file" not in result["output"]:
            raise ValueError("Response output missing 'file'")

        return self.deserialize(result["output"]["file"])

    def to(self, *args, **kwargs):
        if args[0] and args[0] != "cpu":
            warn("`.to()` has no effect on remote models. Ignoring.")
        return self

    def train(self, mode=True):
        if mode:
            raise ValueError("Client cannot be run in training mode.")
        return super().train(mode=False)


if __name__ == "__main__":
    from deepinv.utils import load_image, plot

    # 1. Call real API
    import os
    from dotenv import load_dotenv
    load_dotenv("../ram_docker_demo/.env")
    model = Client(api_key=os.getenv("API_KEY"), endpoint=f"https://api.runpod.ai/v2/{os.getenv('ENDPOINT_ID')}/runsync")
    
    # 2. Call local API served with local runpod
    model = Client(endpoint="http://localhost:8000/runsync")

    # 3. See deepinv/tests/test_models.py:test_client_mocked for unit test calling mocked API

    y = load_image("../ram_docker_demo/butterfly_noisy_tiny.png")

    x_hat = model(y, physics="denoising_from_client")

    plot([y, x_hat])
