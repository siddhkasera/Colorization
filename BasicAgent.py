from PIL.Image import Image
from AutoAgent import AutoAgent


class BasicAgent(AutoAgent):
    """
    BasicAgent

    Executes the basic coloring agent as described by Dr Cowan
    """

    def __init__(self, img: Image, numColors=5):
        super().__init__(img, numColors)

    def execute(self):
        rightWidth, rightLen = self.rightHalf.size
        rgb_rightHalf = self.rightHalf.convert("RGB")
        pass
