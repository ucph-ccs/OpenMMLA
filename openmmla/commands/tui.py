import os


def main():
    os.environ.setdefault("TEXTUAL", "")
    from openmmla.tui.app import OpenMMLAApp
    app = OpenMMLAApp()
    app.run()
