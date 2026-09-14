"""Utilities for selecting which autoregressive prediction terms define the loss."""


def resolve_loss_range(
    num_positions,
    k_steps_for_loss="all",
    loss_positions=None,
):
    """Return a zero-based, half-open ``(start, stop)`` loss range.

    ``loss_positions`` accepts ``all``, ``last:N``, or ``range:START:STOP``.
    The legacy ``k_steps_for_loss`` setting remains supported.
    """
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")

    explicit_spec = None if loss_positions in (None, "") else str(loss_positions)
    legacy_spec = str(k_steps_for_loss)
    if explicit_spec is not None and legacy_spec != "all":
        raise ValueError(
            "Set either training.loss_positions or training.k_steps_for_loss, not both"
        )

    spec = explicit_spec or legacy_spec
    if spec == "all":
        return 0, num_positions

    if spec.startswith("last:"):
        count_text = spec.split(":", 1)[1]
    elif explicit_spec is None:
        count_text = spec
    else:
        count_text = None

    if count_text is not None:
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"Invalid trailing loss count in {spec!r}") from exc
        if count <= 0:
            raise ValueError("The trailing loss count must be positive")
        return max(num_positions - count, 0), num_positions

    if spec.startswith("range:"):
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Loss ranges must use range:START:STOP (zero-based, STOP exclusive)"
            )
        try:
            start, stop = int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ValueError(f"Invalid loss range {spec!r}") from exc
        if not 0 <= start < stop <= num_positions:
            raise ValueError(
                f"Loss range {spec!r} is invalid for {num_positions} positions"
            )
        return start, stop

    raise ValueError(
        f"Unknown loss position specification {spec!r}; expected all, last:N, "
        "range:START:STOP, or the legacy integer form"
    )
