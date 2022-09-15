import csv
from datetime import datetime, timedelta

class CSVMeter:
    def __init__(self, path, buffering=-1):
        self.file_pointer = open(path, 'wt', buffering=buffering)
        self.dict_writer = None
        self.start_time = datetime.now()

    def elapsed_seconds(self):
        diff = datetime.now() - self.start_time
        return diff.total_seconds()

    def update(self, **kwargs):
        if 'elapsed_seconds' not in kwargs:
            kwargs['elapsed_seconds'] = self.elapsed_seconds()

        if self.dict_writer is None:
            self.dict_writer = csv.DictWriter(
                self.file_pointer,
                sorted(list(kwargs.keys())),
            )
            self.dict_writer.writeheader()

        self.dict_writer.writerow(kwargs)

    def flush(self):
        self.file_pointer.flush()

    @classmethod
    def from_csv(cls, oldcsv, newcsv, **kwargs):
        """
        Initialize a meter from an old CSV.
        """
        c = cls(newcsv, **kwargs)

        # adjust start_time to reflect time since last entry in elapsed_seconds field
        with open(oldcsv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            sec_col = header.index('elapsed_seconds')
            for row in reader:
                last_time = row[sec_col]
                # insert previous row into new csv
                c.update(**dict(zip(header, row)))
        c.flush()
        last_time = float(last_time)
        c.start_time -= timedelta(seconds=last_time)

        return c
