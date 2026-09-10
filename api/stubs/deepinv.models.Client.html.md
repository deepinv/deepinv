# Client

### *class* deepinv.models.Client(endpoint, api_key='', return_metadata=False)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

DeepInverse model API Client.

Perform inference on models hosted in the cloud directly from DeepInverse.

This functionality allows contributors to develop APIs to disseminate their reconstruction models,
without requiring the client user to host the model themselves
or to accurately define their physics. As an API developer, all you have to do is:

* Define your model to take tensors as input and output tensors (like [`deepinv.models.Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor))
* Create a simple API (see below for example)
* Deploy it to the cloud, and distribute the endpoint URL and API keys to anyone who might want to use it!

The user then only needs to define this client, specify the endpoint URL and API key, and pass in an image as a tensor.

#### WARNING
This feature is **experimental**. Its interface and behavior may change
without notice in future releases. Use with caution in production workflows.

<hr />

* **Example:**

```python
import deepinv as dinv
import torch
y = torch.tensor([...]) # Your measurements

model = dinv.models.Client("<ENDPOINT>", "<API_KEY>")

x_hat = model(y, physics="denoising")
```

<hr />

**Create your own API**: In order to develop an API to be compatible with this client:

* Since we cannot pass objects via the API, physics are passed as strings with optional parameters and must be rebuilt in the API.
* The API must accept the following input body:

```python
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
```

* The API must return the following output response:

```python
{
    "output": {
        "file": "<b64 serialized file>",
        "metadata": {
            "other_outputs": "such as inference time",
        }
    }
}
```

During forward pass, the client passes input tensor serialized as base64 to API, along with any optional params,
which must either be plain text, numbers, or serializable, depending on the API input requirements,
such as `physics` string, `config`, `sigma`, `mask` etc.

The API can be developed and deployed on any platform you prefer, e.g. server, containers, or functions. See below for
some simple examples.

#### NOTE
Authentication is handled at the application level via the API key by default.
However, you may also choose to enforce authentication or rate-limiting at an upstream
layer (e.g. an nginx reverse proxy or API gateway) if preferred.

#### WARNING
Security is critical when exposing models via Web APIs.
Always use HTTPS, validate and sanitize inputs, and restrict access with strong API keys or
authentication mechanisms. Consider rate-limiting and monitoring to reduce attack surface.

* **Example:**

**Simple server using Flask**

```python
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
```

**Serverless container using RunPod**

```python
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
```

* **Parameters:**
  * **endpoint** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – endpoint URL.
  * **api_key** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – API key.
  * **return_metadata** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – optionally return metadata dict outputted from API.

#### *static* deserialize(data)

Helper function to deserialize client outputs.

The media type for the pickled
documents is expected to be `application/octet-stream`.

* **Parameters:**
  **data** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – input serialized using [`serialize()`](#deepinv.models.Client.serialize)
* **Returns:**
  torch.Tensor deserialized Tensor
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(y, \*\*kwargs)

Client model forward pass.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements tensor
  * **kwargs** – any optional params depending on the API input requirements e.g. `physics` string, `config`, `sigma`, `mask` etc.
* **Returns:**
  torch.Tensor output reconstruction tensor
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* serialize(tensor)

Helper function to serialize client inputs.

Instances of torch.Tensor are serialized by first [pickling](https://docs.python.org/3/library/pickle.html) them using
[`torch.save()`](https://docs.pytorch.org/docs/stable/generated/torch.save.html#torch.save) and then returning a URI pointing to the pickle
file. For now, only data URIs are supported, but in the future
short-lived URLs may also be supported.

* **Parameters:**
  **tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor
* **Returns:**
  tensor serialized as base64 string
* **Return type:**
  [str](https://docs.python.org/3.9/library/stdtypes.html#str)
