"""
Bypasses Hydra's @hydra_runner decorator and whatever the OneLogger telemetry
wrapper does around it, to test whether _generate_session's raise Exception
actually fires when the simulator is instantiated and called in the most
minimal, unadorned way possible -- no decorators, no error-handling layers,
nothing between this code and the real class.

Run from your repo root:
    python bypass_hydra_test.py

If this DOES raise: something in Hydra/OneLogger/@hydra_runner was silently
swallowing the exception in the normal run path -- that's the answer.

If this does NOT raise either: something even stranger is going on, and the
next step is re-examining whether MultiSpeakerSimulator itself is being
subclassed or monkey-patched somewhere else in the codebase.
"""
import traceback

from hydra import compose, initialize
from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator

with initialize(version_base=None, config_path="tools/speech_data_simulator/conf"):
    cfg = compose(
        config_name="data_simulator.yaml",
        overrides=[
            "data_simulator.manifest_filepath=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignment_manifest_clean.json",
            "data_simulator.outputs.output_dir=/tmp/bypass_test_sessions",
            "data_simulator.session_config.num_speakers=2",
            "data_simulator.session_config.num_sessions=10",
            "data_simulator.session_config.session_length=20",
        ],
    )

try:
    simulator = MultiSpeakerSimulator(cfg=cfg)
    simulator.generate_sessions()
    print("\n>>> Completed with NO exception. Genuinely unexpected given the raise -- "
          "means something even stranger than Hydra/OneLogger error-swallowing is going on.")
except Exception:
    print("\n>>> Exception WAS raised here. Something in the normal Hydra/@hydra_runner "
          "path was swallowing it before.")
    traceback.print_exc()
