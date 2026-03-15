"""
TTT adapter.

Design idea:
- run K adaptation steps on test batches,
- update only allowed parameters,
- then produce final prediction.
"""


class TestTimeAdapter:
    # TODO: Prepare adaptation policy (layer scope, steps, lr).
    def __init__(self) -> None:
        pass

    # TODO: Run adaptation on a test batch and return updated state.
    def adapt(self, batch):
        pass
