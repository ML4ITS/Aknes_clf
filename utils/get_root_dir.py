from pathlib import Path


def get_root_dir():
    return Path(__file__).parent.parent


root_dir = get_root_dir()


if __name__ == '__main__':
    print(root_dir)
    print(root_dir.joinpath('some_dir1', 'some_dir2'))