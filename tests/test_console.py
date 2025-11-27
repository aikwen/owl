from owl.utils import console

if __name__ == "__main__":
    console.welcome()
    console.highlight("file log is {}", "train_fine.log", with_prefix=True)
    console.stop()