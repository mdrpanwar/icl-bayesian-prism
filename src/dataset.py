from torch.utils.data import Dataset


class InContextDataset(Dataset):
    def __init__(self, task_sampler, data_sampler, n_points, n_dims, fake_len=5000):
        self.task_sampler = task_sampler
        self.data_sampler = data_sampler
        self.fake_len = fake_len
        self.n_points = n_points
        self.n_dims = n_dims

    def __getitem__(self, idx):
        task = self.task_sampler()
        xs = self.data_sampler.sample_xs(
            self.n_points,
            1,
            self.n_dims,
        )
        ys = task.evaluate(xs)

        return {"xs": xs[0], "ys": ys[0]}

    def __len__(self):
        return self.fake_len
