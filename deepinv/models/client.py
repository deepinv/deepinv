import torch
import requests
import io
import base64
import json
from warnings import warn
from typing import Any

from deepinv.models import Reconstructor, Denoiser


class Client(Reconstructor, Denoiser):
    r"""
    DeepInverse model API Client.

    Perform inference on models hosted in the cloud directly from DeepInverse.

    The client allows contributors to disseminate their reconstruction models, without requiring the user to have high GPU resources
    or to accurately define their physics. As a contributor, all you have to do is:

    * Define your model to take tensors as input and output tensors (like :class:`deepinv.models.Reconstructor`)
    * Create a simple API (see below for example)
    * Deploy it to the cloud, and distribute the endpoint URL and API keys to anyone who might want to use it!

    The user then only needs to define this client, specify the endpoint URL and API key, and pass in an image as a tensor.

    |sep|

    :Example:

    ::

        import deepinv as dinv
        import torch
        y = torch.tensor([...]) # Your measurements

        model = dinv.models.Client("<ENDPOINT>", "<API_KEY>")

        x_hat = model(y, physics="denoising")

    |sep|

    **Create your own API**: In order to develop an API to be compatible with this client:

    * Since we cannot pass objects via the API, physics are passed as strings with optional parameters and must be rebuilt in the API.
    * The API must accept the following input body:

    ```python
    {
        "input": {
            "file": <b64 serialized file>,
            "param1": "such as a config str",
            "param2": <or a b64 serialized param>,
            ...
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

    During forward pass, the client passes input tensor serialized as base64 to API, along with any optional params,
    which must either be plain text, numbers, or serializable, depending on the API input requirements,
    such as `physics` string, `config`, `sigma`, `mask` etc.

    :Example:

    **Simple server using Flask** ::

        from flask import Flask, request, jsonify
        from deepinv.models import Client

        app = Flask(__name__)
        model = ... # Your DeepInverse model

        @app.route("/", methods=["POST"])
        def infer():
            inp = request.get_json()["input"]
            y = Client.deserialize(inp["file"])
            physics = ... # Create physics depending on other params in inp

            x_hat = model(y, physics) # Server-side inference

            return jsonify({
                "output": {
                    "file": Client.serialize(x_hat)
                }
            })

        if __name__ == "__main__":
            app.run()

    **Server using RunPod** ::

        import runpod
        from deepinv.models import Client

        model = ... # Your DeepInverse model

        def handler(event):
            inp = event['input']
            y = Client.deserialize(inp["file"])
            physics = ... # Create physics depending on other params in inp

            x_hat = model(y, physics) # Server-side inference

            return {
                "output": {
                    "file": Client.serialize(x_hat)
                }
            }

        if __name__ == '__main__':
            runpod.serverless.start({'handler': handler })

    :param str endpoint: endpoint URL.
    :param str api_key: API key.
    """

    def __init__(self, endpoint: str, api_key: str = ""):
        super().__init__(device=None)
        self.api_key = api_key
        self.endpoint = endpoint
        self.training = False

    @staticmethod
    def serialize(tensor: torch.Tensor) -> str:
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def deserialize(data: str) -> torch.Tensor:
        buffer = io.BytesIO(base64.b64decode(data))
        return torch.load(buffer, map_location="cpu")

    @staticmethod
    def _check_value(v: Any):
        """
        Checks if value can be directly jsonified, or serialises tensors, otherwise raise error.
        """
        ALLOWED = (int, float, str, bool, type(None))
        if isinstance(v, torch.Tensor):
            return Client.serialize(v)
        elif isinstance(v, ALLOWED):
            return v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, ALLOWED) for x in v):
                return v
            raise TypeError("Lists/tuples may only contain primitive types")

        raise TypeError(f"Unsupported kwarg value type: {type(v).__name__}")

    def forward(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Client model forward pass.

        :param torch.Tensor y: input measurements tensor
        :param kwargs: any optional params depending on the API input requirements e.g. `physics` string, `config`, `sigma`, `mask` etc.
        :return: torch.Tensor output reconstruction tensor
        """
        if self.training:
            raise RuntimeError("Model client can only be used in evaluation mode.")

        safe_kwargs = {
            k: Client._check_value(v) for k, v in kwargs.items() if isinstance(k, str)
        }
        if len(safe_kwargs) != len(kwargs):
            raise TypeError("All kwarg keys must be strings")

        payload = {"input": {"file": Client.serialize(y), **safe_kwargs}}

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

        return Client.deserialize(result["output"]["file"])

    def to(self, *args, **kwargs):
        if args[0] and args[0] != "cpu":
            warn("`.to()` has no effect on remote models. Ignoring.")
        return self

    def train(self, mode=True):
        if mode:
            raise ValueError("Client cannot be run in training mode.")
        return super().train(mode=False)
