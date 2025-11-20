from owl.utils import file_io

if __name__ == '__main__':

    f1 = file_io.create_logger("2025-11-19","train")
    f2 = file_io.create_logger("2025-11-19", "train")

    f1.info("abc1")
    f2.info("abc2")
    f1.warning("abc3")

