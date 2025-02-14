def _read_hebo(file_path: str = "./hebo.txt") -> str:
    with open(file_path, "r") as file:
        return file.read()


HEBO = _read_hebo()
