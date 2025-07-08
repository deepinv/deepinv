# Security Policy

## Supported Versions

Only the latest stable release of DeepInverse is officially supported with security updates.

| Version        | Supported |
|----------------|-----------|
| `stable`       | ✅        |
| `< stable`     | ❌        |


## Reporting a Vulnerability

If you believe you have found a security vulnerability in DeepInverse, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem. Please report issues by contacting a [lead developer](https://deepinv.github.io/).

## Using DeepInverse securely

While DeepInverse is designed for research purposes, users should take care to use it securely:

- **Automatic downloads**: Some features in DeepInverse automatically download weights, datasets or images from the internet, including our [DeepInverse HuggingFace](https://huggingface.co/deepinv/) repositories and other external sources. If running DeepInverse in sensitive or production environments, always verify the source and contents of downloaded assets beforehand, or consider disabling network access or controlling outbound connections to limit unexpected downloads or data leaks.

- **Trusted code**: Avoid executing modified or third-party scripts that haven't been reviewed. Only use models and datasets from sources you trust. Run DeepInverse in a virtual environment, container or sandbox to isolate it from the rest of your system.

- **Dependencies**: Ensure DeepInverse and all its dependencies are regularly updated to include security patches.

- **Confidential data**: DeepInverse is not currently designed with data privacy or secure deployment guarantees. Do not process sensitive or proprietary information without a proper security review.