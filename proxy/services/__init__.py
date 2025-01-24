def _read_gato(file_path: str = "../hebo.txt") -> str:
    with open(file_path, "r") as file:
        return file.read()


HEBO = _read_gato()
