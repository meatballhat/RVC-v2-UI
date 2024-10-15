import os
import sys
import urllib.parse
from argparse import Namespace
from cog import BasePredictor, Input, Path as CogPath

sys.path.insert(0, os.path.abspath("src"))

import main as m

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        input_audio: CogPath = Input(
            description="Upload your audio file here.",
            default=None,
        ),
        rvc_model: str = Input(
            description="RVC model for a specific voice. If using a custom model, this should match the name of the downloaded model. If a 'custom_rvc_model_download_url' is provided, this will be automatically set to the name of the downloaded model.",
            default="Obama",
            choices=[
                "Obama",
                "Trump",
                "Sandy",
                "Rogan",
                "CUSTOM",
            ],
        ),
        custom_rvc_model_download_url: str = Input(
            description="URL to download a custom RVC model. If provided, the model will be downloaded (if it doesn't already exist) and used for prediction, regardless of the 'rvc_model' value.",
            default=None,
        ),
        pitch_change: float = Input(
            description="Adjust pitch of AI vocals in semitones. Use positive values to increase pitch, negative to decrease.",
            default=0,
        ),
        index_rate: float = Input(
            description="Control how much of the AI's accent to leave in the vocals.",
            default=0.5,
            ge=0,
            le=1,
        ),
        filter_radius: int = Input(
            description="If >=3: apply median filtering to the harvested pitch results.",
            default=3,
            ge=0,
            le=7,
        ),
        rms_mix_rate: float = Input(
            description="Control how much to use the original vocal's loudness (0) or a fixed loudness (1).",
            default=0.25,
            ge=0,
            le=1,
        ),
        f0_method: str = Input(
            description="Pitch detection algorithm. 'rmvpe' for clarity in vocals, 'mangio-crepe' for smoother vocals.",
            default="rmvpe",
            choices=["rmvpe", "mangio-crepe"],
        ),
        crepe_hop_length: int = Input(
            description="When `f0_method` is set to `mangio-crepe`, this controls how often it checks for pitch changes in milliseconds.",
            default=128,
        ),
        protect: float = Input(
            description="Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable.",
            default=0.33,
            ge=0,
            le=0.5,
        ),
        output_format: str = Input(
            description="wav for best quality and large file size, mp3 for decent quality and small file size.",
            default="mp3",
            choices=["mp3", "wav"],
        ),
    ) -> CogPath:
        """
        Runs a single prediction on the model.
        """
        if custom_rvc_model_download_url:
            custom_rvc_model_download_name = urllib.parse.unquote(
                custom_rvc_model_download_url.split("/")[-1]
            )
            custom_rvc_model_download_name = os.path.splitext(
                custom_rvc_model_download_name
            )[0]
            print(
                f"[!] The model will be downloaded as '{custom_rvc_model_download_name}'."
            )
            m.download_online_model(
                url=custom_rvc_model_download_url,
                dir_name=custom_rvc_model_download_name,
                overwrite=True
            )
            rvc_model = custom_rvc_model_download_name
        else:
            print(
                "[!] Since no URL was provided, we will use the selected RVC model."
            )

        rvc_dirname = rvc_model
        if not os.path.exists(os.path.join(m.rvc_models_dir, rvc_dirname)):
            raise Exception(
                f"The folder {os.path.join(m.rvc_models_dir, rvc_dirname)} does not exist."
            )

        output_path = m.voice_conversion(
            str(input_audio),
            rvc_dirname,
            pitch_change,
            f0_method,
            index_rate,
            filter_radius,
            rms_mix_rate,
            protect
        )
        print(f"[+] Converted audio generated at {output_path}")

        # Return the output path
        return CogPath(output_path)
