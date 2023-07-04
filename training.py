from functools import wraps

def record(m, obj):
  """helper function that records activations from down blocks"""
  _m = m.forward

  @wraps(m.forward)
  def _f(*args, **kwargs):
    res = _m(*args, **kwargs)
    obj.saved.append(res)
    return res

  m.forward = _f
  return m
