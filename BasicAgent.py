from PIL.Image import Image
from AutoAgent import AutoAgent


class BasicAgent(AutoAgent):
    """
    BasicAgent

    Executes the basic coloring agent as described by Dr Cowan
    """

    def __init__(self, img: Image):
        super().__init__(img)

    # TODO
