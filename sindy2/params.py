from dataclasses import dataclass

@dataclass
class Params:
    """Dataclass to hold parameters for the physical system and learning process."""
    m1: float  # Mass 1
    m2: float  # Mass 2
    k1: float  # Spring constant 1
    k2: float  # Spring constant 2
    c1: float  # Damping coefficient 1
    c2: float  # Damping coefficient 2
    fr1: dict  # Friction parameters for mass 1
    fr2: dict  # Friction parameters for mass 2
    F1: float  # Forcing amplitude 1
    freq1: float  # Forcing frequency 1
    F2: float  # Forcing amplitude 2
    freq2: float  # Forcing frequency 2
    phi: float  # Phase shift for forcing 2
    x0: list  # Initial conditions [x1, x2, v1, v2]
    timefinal: float  # Final time
    timestep: float  # Time step
    poly_order: int  # Polynomial order for features
    cos_phases: list  # Cosine phase coefficients
    sin_phases: list  # Sine phase coefficients (if any)
    y1_sgn_flag: bool  # Signum feature flag for y1
    y2_sgn_flag: bool  # Signum feature flag for y2
    x_sgn_flag: bool  # Signum feature flag for x (if any)
    y_sgn_flag: bool  # Signum feature flag for y (if any)
    log_1_fr1: bool  # Log feature flag 1 for fr1
    log_2_fr1: bool  # Log feature flag 2 for fr1
    log_1_fr2: bool  # Log feature flag 1 for fr2
    log_2_fr2: bool  # Log feature flag 2 for fr2
    lr: float  # Learning rate
    weightdecay: float  # Weight decay for optimization
    num_iter: int  # Number of training iterations
    num_epochs: int  # Number of epochs per iteration
    mus: list = None  # Means for scaling (optional)
    stds: list = None  # Std devs for scaling (optional)