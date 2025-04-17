from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()

class DrowningDetectionCNN:
    def __init__(self):
        pass

    def predict(self, path: str):
        CLIENT = InferenceHTTPClient(
        api_url=os.environ.get("API_URL"),
        api_key=os.environ.get("API_KEY")
        )

        result = CLIENT.infer(path, model_id=os.environ.get("MODEL_ID"))

        return {"image": result["image"], "predictions": result["predictions"][0]}
