from cog import BasePredictor, Input

class Predictor(BasePredictor):

    def setup(self):
        return

    def predict(
        self,
        prompt: str = Input(description="The prompt", default="catgirl, cat ears, white hair, golden eyes, bob cut, pov, face closeup, smile"),
    ) -> str:
        return prompt