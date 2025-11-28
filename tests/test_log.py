from owl.utils import file_io

if __name__ == '__main__':

    f1 = file_io.get_logger("2025-11-19", "train")
    f2 = file_io.get_logger("2025-11-19", "train")

    f1.info("abc1")
    f2.info("abc2")
    f1.warning("abc3")

    val_log1 = file_io.get_logger("2025-11-19", "val", is_format=False)
    val_log2 = file_io.get_logger("2025-11-19", "val")

    val_log1.info("abc1")
    val_log2.info("abc2")
    
