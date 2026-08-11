from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from owl.data.samper.balanced import BalancedSampler


class StudentDataset(Dataset):
    """A simple class roster dataset."""

    def __init__(self, class_name: str, student_count: int) -> None:
        self.class_name = class_name
        self.students = [
            f"{class_name}-{index + 1}"
            for index in range(student_count)
        ]

    def __len__(self) -> int:
        return len(self.students)

    def __getitem__(self, index: int) -> str:
        return self.students[index]

if __name__ == "__main__":
    class_a = StudentDataset("Class-A", 8)
    class_b = StudentDataset("Class-B", 8)
    class_c = StudentDataset("Class-C", 8)

    dataset = ConcatDataset([
        class_a,
        class_b,
        class_c,
    ])

    sampler = BalancedSampler(
        dataset,
        samples_per_dataset=3,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
    )

    print("Class sizes:")
    print("A:", len(class_a))
    print("B:", len(class_b))
    print("C:", len(class_c))
    print()

    print("ConcatDataset cumulative sizes:")
    print(dataset.cumulative_sizes)
    print()

    for epoch in range(1, 4):
        print(f"========== Epoch {epoch} ==========")

        selected = {
            "Class-A": [],
            "Class-B": [],
            "Class-C": [],
        }

        print("Lottery order:")

        for batch_index, student_batch in enumerate(loader, start=1):
            student = student_batch[0]

            print(
                f"{batch_index:02d}: {student}"
            )

            class_name = student.rsplit("-", 1)[0]
            selected[class_name].append(student)

        print()
        print("Selected students by class:")

        for class_name, students in selected.items():
            print(
                f"{class_name}: {students}"
            )

        print()
