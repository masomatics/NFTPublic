from PIL import ImageStat
import numpy as np
import torchvision.transforms.functional as TF

class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, self.h, other.h)))

# loader = DataLoader(dataset, batch_size=100, num_workers=8)

def data_stats(loader):
    statistics = None
    for (data, _), _, _ in loader:
        for b in range(data.shape[0]):
            if statistics is None:
                statistics = Stats(TF.to_pil_image(data[b]))
            else:
                statistics += Stats(TF.to_pil_image(data[b]))
    return statistics
