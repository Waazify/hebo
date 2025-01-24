import uuid


def generate_id() -> uuid.UUID:
    """Generate a unique ID."""
    return uuid.uuid4()
