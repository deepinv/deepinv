# DownloadError

### *class* deepinv.utils.DownloadError(\*args, \*\*kwargs)

Bases: [`RequestException`](https://docs.python-requests.org/en/latest/api/#requests.RequestException)

Raised when a network download initiated by deepinv fails.

Wraps any underlying network error (HTTP status, connection failure,
SSL error, DNS failure, timeout, …) so that callers — and the test
suite in particular — can detect download failures with a single
`except DownloadError:` rather than enumerating every exception
type that the network stack may raise. The original exception is
chained via `__cause__`.

Inherits from [`requests.exceptions.RequestException`](https://docs.python-requests.org/en/latest/api/#requests.RequestException) (and
transitively from `IOError`) so existing handlers that catch
those broader types keep working.
