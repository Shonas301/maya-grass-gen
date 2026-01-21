"""opensimplex fBm wrapper for generative art.

drop-in replacement for noise.pnoise2/pnoise3 with octave support.
"""

import opensimplex


def init_noise(seed: int | None = None) -> None:
    """Initialize noise generator with seed.

    Args:
        seed: random seed (uses default if None)
    """
    if seed is not None:
        opensimplex.seed(seed)


def fbm_noise2(
    x: float,
    y: float,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> float:
    """Generate 2D fractal Brownian motion noise.

    drop-in replacement for noise.pnoise2().

    Args:
        x: x coordinate
        y: y coordinate
        octaves: number of noise layers (default 1)
        persistence: amplitude multiplier per octave (default 0.5)
        lacunarity: frequency multiplier per octave (default 2.0)

    Returns:
        noise value in range [-1.0, 1.0]
    """
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        total += opensimplex.noise2(x * frequency, y * frequency) * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_amplitude


def fbm_noise3(
    x: float,
    y: float,
    z: float,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> float:
    """Generate 3D fractal Brownian motion noise.

    drop-in replacement for noise.pnoise3().

    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate (use for time-based animation)
        octaves: number of noise layers (default 1)
        persistence: amplitude multiplier per octave (default 0.5)
        lacunarity: frequency multiplier per octave (default 2.0)

    Returns:
        noise value in range [-1.0, 1.0]
    """
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        total += (
            opensimplex.noise3(x * frequency, y * frequency, z * frequency) * amplitude
        )
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_amplitude


def fbm_noise1(
    x: float,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> float:
    """Generate 1D fractal Brownian motion noise.

    drop-in replacement for noise.pnoise1().
    implemented as noise2(x, 0) since opensimplex has no noise1.

    Args:
        x: x coordinate
        octaves: number of noise layers (default 1)
        persistence: amplitude multiplier per octave (default 0.5)
        lacunarity: frequency multiplier per octave (default 2.0)

    Returns:
        noise value in range [-1.0, 1.0]
    """
    return fbm_noise2(x, 0.0, octaves, persistence, lacunarity)
