import os


class FolderCreator:

    @staticmethod
    def create_folder(path) -> None:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created Directory at: {path}")
