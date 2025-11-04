from typing import Optional
import torch
from torch.utils.data import Sampler


class BucketSampler(Sampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        boundaries: list[int],
        shuffle: bool = True,
    ):
        self.lengths = [len(item) for item in dataset]
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.shuffle = shuffle

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(self.lengths):
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        # Remove empty buckets
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for bucket in buckets:
            len_bucket = len(bucket)
            rem = (self.batch_size - (len_bucket % self.batch_size)) % self.batch_size
            num_samples_per_bucket.append(len_bucket + rem)

        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(torch.initial_seed())

        indices = []
        for bucket in self.buckets:
            if self.shuffle:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
            else:
                indices.append(list(range(len(bucket))))

        batches = []
        for i, bucket in enumerate(self.buckets):
            ids_bucket = indices[i]
            len_bucket = len(bucket)
            num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket

            # Pad bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # batching
            for j in range(0, len(ids_bucket), self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j : j + self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]

        return iter(batches)

    def _bisect(self, x: int, lo: int = 0, hi: Optional[int] = None) -> int:
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
