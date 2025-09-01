import requests
import io
import base64
import json
from warnings import warn
from typing import Any
from urllib.request import UnknownHandler, DataHandler, OpenerDirector

import torch
from deepinv.models import Reconstructor, Denoiser


class Client(Reconstructor, Denoiser):
    r"""
    DeepInverse model API Client.

    Perform inference on models hosted in the cloud directly from DeepInverse.

    This functionality allows contributors to develop APIs to disseminate their reconstruction models,
    without requiring the client user to host the model themselves
    or to accurately define their physics. As an API developer, all you have to do is:

    * Define your model to take tensors as input and output tensors (like :class:`deepinv.models.Reconstructor`)
    * Create a simple API (see below for example)
    * Deploy it to the cloud, and distribute the endpoint URL and API keys to anyone who might want to use it!

    The user then only needs to define this client, specify the endpoint URL and API key, and pass in an image as a tensor.

    .. warning::

        This feature is **experimental**. Its interface and behavior may change
        without notice in future releases. Use with caution in production workflows.

    |sep|

    :Example:

    .. code-block:: python

        import deepinv as dinv
        import torch
        y = torch.tensor([...]) # Your measurements

        model = dinv.models.Client("<ENDPOINT>", "<API_KEY>")

        x_hat = model(y, physics="denoising")

    |sep|

    **Create your own API**: In order to develop an API to be compatible with this client:

    * Since we cannot pass objects via the API, physics are passed as strings with optional parameters and must be rebuilt in the API.
    * The API must accept the following input body:

    .. code-block:: python

        {
            "input": {
                "file": <b64 serialized file>,
                "metadata": {
                    "param1": "such as a config str",
                    "param2": <or a b64 serialized param>,
                    ...
                },
            }
        }

    * The API must return the following output response:

    .. code-block:: python

        {
            "output": {
                "file": "<b64 serialized file>",
                "metadata": {
                    "other_outputs": "such as inference time",
                }
            }
        }

    During forward pass, the client passes input tensor serialized as base64 to API, along with any optional params,
    which must either be plain text, numbers, or serializable, depending on the API input requirements,
    such as `physics` string, `config`, `sigma`, `mask` etc.

    The API can be developed and deployed on any platform you prefer, e.g. server, containers, or functions. See below for
    some simple examples.

    .. note::
        Authentication is handled at the application level via the API key by default.
        However, you may also choose to enforce authentication or rate-limiting at an upstream
        layer (e.g. an nginx reverse proxy or API gateway) if preferred.

    .. warning::
        Security is critical when exposing models via Web APIs.
        Always use HTTPS, validate and sanitize inputs, and restrict access with strong API keys or
        authentication mechanisms. Consider rate-limiting and monitoring to reduce attack surface.

    :Example:

    **Simple server using Flask**

    .. code-block:: python

        from flask import Flask, request, jsonify
        from deepinv.models import Client

        app = Flask(__name__)
        model = ... # Your DeepInverse model

        @app.route("/", methods=["POST"])
        def infer():
            inp = request.get_json()["input"]
            y = Client.deserialize(inp["file"])
            physics = ... # Create physics depending on metadata

            x_hat = model(y, physics) # Server-side inference

            return jsonify({
                "output": {
                    "file": Client.serialize(x_hat)
                }
            })

        if __name__ == "__main__":
            app.run()

    **Serverless container using RunPod**

    .. code-block:: python

        import runpod
        from deepinv.models import Client

        model = ... # Your DeepInverse model

        def handler(event):
            inp = event['input']
            y = Client.deserialize(inp["file"])
            physics = ... # Create physics depending on metadata

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
    :param bool return_metadata: optionally return metadata dict outputted from API.
    """

    def __init__(self, endpoint: str, api_key: str = "", return_metadata: bool = False):
        super().__init__(device=None)
        self.api_key = api_key
        self.endpoint = endpoint
        self.training = False
        self.return_metadata = return_metadata

    @staticmethod
    def serialize(tensor: torch.Tensor) -> str:
        """Helper function to serialize client inputs.

        Instances of torch.Tensor are serialized by first `pickling
        <https://docs.python.org/3/library/pickle.html>`_ them using
        :func:`torch.save` and then returning a URI pointing to the pickle
        file. For now, only data URIs are supported, but in the future
        short-lived URLs may also be supported.

        :param torch.Tensor tensor: input tensor
        :return: tensor serialized as base64 string
        """
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:application/octet-stream;base64,{b64}"

    @staticmethod
    def deserialize(data: str) -> torch.Tensor:
        """
        Helper function to deserialize client outputs.

        The media type for the pickled
        documents is expected to be ``application/octet-stream``.

        :param str data: input serialized using :meth:`serialize`
        :return: torch.Tensor deserialized Tensor
        """
        opener = OpenerDirector()
        for handler in [
            DataHandler(),  # Data URIs
            UnknownHandler(),  # Fallback
        ]:
            opener.add_handler(handler)

        with opener.open(data) as f:
            ctype = f.headers.get_content_type()

            if ctype != "application/octet-stream":
                raise RuntimeError(
                    f"Unexpected media type: {ctype}, expected 'application/octet-stream'"
                )

            obj = torch.load(f, map_location="cpu", weights_only=True)

            if not isinstance(obj, torch.Tensor):
                raise RuntimeError(f"Expected a torch.Tensor, got {type(obj).__name__}")

            return obj

    @staticmethod
    def _sanitize_value(v: Any):
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

        params = {k: Client._sanitize_value(v) for k, v in kwargs.items()}

        payload = {"input": {"file": Client.serialize(y), "metadata": params}}

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

        if "output" not in result or not isinstance(result["output"], dict):
            raise ValueError("Response missing 'output' field")

        if "file" not in result["output"]:
            raise ValueError("Response output missing 'file'")

        out = Client.deserialize(result["output"]["file"])

        if self.return_metadata:
            metadata = result["output"].get("metadata")

            if metadata is not None:
                metadata = {k: Client._sanitize_value(v) for k, v in metadata.items()}

            return out, metadata
        else:
            return out

    def to(self, *args, **kwargs):
        if args[0] and args[0] != "cpu":
            warn("`.to()` has no effect on remote models. Ignoring.")
        return self

    def train(self, mode=True):
        if mode:
            raise ValueError("Client cannot be run in training mode.")
        return super().train(mode=False)
