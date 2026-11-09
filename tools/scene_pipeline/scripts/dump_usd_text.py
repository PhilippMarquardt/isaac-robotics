import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
from pxr import Usd
stage = Usd.Stage.Open(args.usd)
with open(args.usd + ".usda.txt", "w", encoding="utf-8") as f:
    f.write(stage.GetRootLayer().ExportToString())
print("WROTE", args.usd + ".usda.txt")
simulation_app.close()
