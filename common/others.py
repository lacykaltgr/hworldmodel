
# reward stream normalization
class StreamNorm(tfutils.Module):

  def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
    # Momentum of 0 normalizes only based on the current batch.
    # Momentum of 1 disables normalization.
    self._shape = tuple(shape)
    self._momentum = momentum
    self._scale = scale
    self._eps = eps
    self.mag = tf.Variable(tf.ones(shape, tf.float64), False)

  def __call__(self, inputs):
    metrics = {}
    self.update(inputs)
    metrics['mean'] = inputs.mean()
    metrics['std'] = inputs.std()
    outputs = self.transform(inputs)
    metrics['normed_mean'] = outputs.mean()
    metrics['normed_std'] = outputs.std()
    return outputs, metrics

  def reset(self):
    self.mag.assign(tf.ones_like(self.mag))

  def update(self, inputs):
    batch = inputs.reshape((-1,) + self._shape)
    mag = tf.abs(batch).mean(0).astype(tf.float64)
    self.mag.assign(self._momentum * self.mag + (1 - self._momentum) * mag)

  def transform(self, inputs):
    values = inputs.reshape((-1,) + self._shape)
    values /= self.mag.astype(inputs.dtype)[None] + self._eps
    values *= self._scale
    return values.reshape(inputs.shape)