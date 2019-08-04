try:
    from tqdm import tqdm
    TQDM_ENABLED = True
except ImportError:
    TQDM_ENABLED = False


class TorchLogger:
    def __init__(self, total_epoch, total_step, summary_writer=None):
        self.total_epoch = total_epoch
        self.total_step = total_step
        self.writer = summary_writer

        self.sum_dict = {}
        self.count_dict = {}
        self.avg_dict = {}

    def reset(self):
        self.sum_dict = {}
        self.count_dict = {}
        self.avg_dict = {}

    def log(self, epoch, step, **kwargs):
        """
        logger.log(loss=loss.item(), acc=acc, ...)
        :param kwargs:
        :return:
        """
        self.update(**kwargs)
        print(self.get_log_string(epoch, step))

    def update(self, **kwargs):
        """
        avg_loss, avg_acc, ... = logger.update(loss=loss.item(), acc=acc, ...)
        :param kwargs:
        :return:
        """
        value_dict = kwargs

        for key, value in value_dict.items():
            if key in self.sum_dict:
                self.sum_dict[key] += value
            else:
                self.sum_dict[key] = value
            if key in self.count_dict:
                self.count_dict[key] += 1
            else:
                self.count_dict[key] = 1

        for key, value in self.sum_dict.items():
            if self.count_dict[key] == 0:
                self.count_dict[key] = 1
            self.avg_dict[key] = self.sum_dict[key] / self.count_dict[key]

        return self.avg_dict

    def get_log_string(self, epoch, step):
        msg = 'Epoch: [{}/{}]\tStep: [{}/{}]\t'.format(epoch, self.total_epoch, step, self.total_step)

        for i, (key, value) in enumerate(self.avg_dict.items()):
            msg += '{}: {:.3f}'.format(key, value)
            if i+1 != len(self.avg_dict):
                msg += '\t'

        return msg

    def print_log(self, epoch, step):
        print(self.get_log_string(epoch, step))

    def update_tensorboard(self, epoch):
        for i, (key, value) in enumerate(self.avg_dict.items()):
            self.writer.add_scalar(str(key), value, global_step=epoch)
