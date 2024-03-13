import math


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points

        if args.max_freq.start is not None:
            self.max_freq = args.max_freq.start
            self.max_freq_schedule = args.max_freq
        else:
            self.max_freq = None

        if args.rff_dim.start is not None:
            self.rff_dim = args.rff_dim.start
            self.rff_dim_schedule = args.rff_dim
        else:
            self.rff_dim = None
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)
        if self.max_freq is not None:
            self.max_freq = self.update_var(self.max_freq, self.max_freq_schedule)
        if self.rff_dim is not None:
            self.rff_dim = self.update_var(self.rff_dim, self.rff_dim_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
