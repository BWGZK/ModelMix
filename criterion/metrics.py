class Criterion(object):
    def __init__(self):
        super(Criterion, self).__init__()

    def calc_Dice(self, output, target):   
        num = (output * target).sum()
        den = output.sum() + target.sum()
        dice = 2 * num / (den + 1e-5)  
        return dice

    def __call__(self, output, target):

        # C0: 0-0-bg, 1-200-myo, 2-500-lv, 3-1220-edema, 4-2221-scar
        myo_dice = self.calc_Dice((output==1)|(output==3)|(output==4), (target==1)|(output==3)|(output==4)) # myo
        lv_dice = self.calc_Dice((output==2), (target==2)) # lv
        scar_dice = self.calc_Dice((output==4), (target==4)) # scar
        edema_dice = self.calc_Dice((output==3)|(output==4), (target==3)|(target==4)) # edema

        return myo_dice, lv_dice, scar_dice, edema_dice


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.myo = 0.
        self.lv = 0.
        self.scar = 0.
        self.edema = 0.
        self.count = 0
        self.myo_avg = 0.
        self.lv_avg = 0.
        self.scar_avg = 0.
        self.edema_avg = 0.

    def update(self, myo_dice, lv_dice, scar_dice, edema_dice):
        self.myo += myo_dice
        self.lv += lv_dice
        self.scar += scar_dice
        self.edema += edema_dice
        self.count += 1
        self.myo_avg = self.myo / self.count
        self.lv_avg = self.lv / self.count
        self.scar_avg = self.scar / self.count
        self.edema_avg = self.edema / self.count


class Logger(object):
    def __init__(self):
        super(Logger, self).__init__()

    def __call__(self, epoch, total_epoch, file_name, cardiac_dice, pathology_dice):   
        with open(file_name, 'a') as f:
            f.write("=> Epoch: {:0>3d}/{:0>3d} || ".format(epoch, total_epoch))
            f.write("MYO: {:.4f} || ".format(cardiac_dice[0].cpu()))
            f.write("LV: {:.4f} || ".format(cardiac_dice[1].cpu()))
            f.write("Scar: {:.4f} || ".format(pathology_dice[0].cpu()))
            f.write("Scar & Edema: {:.4f}\n".format(pathology_dice[1].cpu()))
            