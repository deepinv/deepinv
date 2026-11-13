<a id="env-variable"></a>

# Environment Variables

The following environment variables can be used to configure the behavior of the `deepinv` library:

#### Environment Variables and Descriptions

| **Name**                   | **Description**                                                                                                                         |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `DEEPINV_CACHE_DIR`        | Set the caching directory for the library, such as download datasets. Set to `"/path/to/cache"`. Default to `"~/.cache/deepinv"`.       |
| `DEEPINV_DOWNLOAD_VERBOSE` | Set the verbosity of model weight download operations. Set to `"1"` to enable verbose output, or `"0"` to disable it. Default is `"1"`. |
