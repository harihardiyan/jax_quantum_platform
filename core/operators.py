import jax.numpy as jnp

sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
identity = jnp.eye(2, dtype=jnp.complex64)
