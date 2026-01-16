import optax
import jax
import jax.numpy as jnp


def make_optimizer(name: str, lr: float):
    """
    Create an Optax optimizer by name.

    Parameters
    ----------
    name : str
        Optimizer name: "adam", "sgd", "rmsprop".
    lr : float
        Learning rate.

    Returns
    -------
    optax.GradientTransformation
        The optimizer object.
    """
    name = name.lower()
    if name == "adam":
        return optax.adam(lr)
    if name == "sgd":
        return optax.sgd(lr)
    if name == "rmsprop":
        return optax.rmsprop(lr)
    raise ValueError(f"Unknown optimizer '{name}'")


def vqe_optimize(cost_fn, params_init, optimizer, steps: int = 200):
    """
    Generic VQE optimization loop.

    Parameters
    ----------
    cost_fn : Callable
        Function mapping params -> cost.
    params_init : jnp.ndarray
        Initial parameters.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    steps : int
        Number of optimization steps.

    Returns
    -------
    params : jnp.ndarray
        Optimized parameters.
    history : list
        Cost history.
    """
    opt_state = optimizer.init(params_init)
    params = params_init

    @jax.jit
    def step(params, opt_state):
        cost, grads = jax.value_and_grad(cost_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, cost

    history = []
    for _ in range(steps):
        params, opt_state, cost = step(params, opt_state)
        history.append(float(cost))

    return params, history
