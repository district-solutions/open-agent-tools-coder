from typing import Dict, Any
import ujson as json

def pp(d: Dict[str, Any] | list[Any] | Any | None = None):
    if d is None:
        return '{}'
    else:
        return json.dumps(d, indent=2, escape_forward_slashes=False)
