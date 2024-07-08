import torch
# from process import LargestConnectedComponents
from criterion.metrics import Criterion, AverageMeter, Logger


@torch.no_grad()
def CrossModalSegNetValidation(args, epoch, backbone, head, valid_image, valid_loader, writer, log_name, curve_name_scar,
                               curve_name_edema):
    logger = Logger()
    criterion = Criterion()
    # keepLCC = LargestConnectedComponents()

    valid_meter = AverageMeter()

    test_C0 = torch.FloatTensor(1, 1, args.dim, args.dim).cuda()
    test_DE = torch.FloatTensor(1, 1, args.dim, args.dim).cuda()
    test_T2 = torch.FloatTensor(1, 1, args.dim, args.dim).cuda()

    test_label = torch.FloatTensor(args.dim, args.dim).cuda()

    for _ in range(int(len(valid_image))):
        C0_Image, DE_Image, T2_Image, label, _ = next(valid_loader)

        test_C0.copy_(C0_Image)
        test_DE.copy_(DE_Image)
        test_T2.copy_(T2_Image)

        input = torch.cat([test_C0, test_DE, test_T2],1)
        res = head(backbone(input))
        res = res["seg"]

        seg = torch.argmax(res, dim=1).squeeze(0)
        # seg = keepLCC(seg.cpu()).cuda()

        test_label.copy_(label[0, 0, ...])

        myo, lv, scar, edema = criterion(seg, test_label)

        valid_meter.update(myo, lv, scar, edema)

    cardiac_dice = [valid_meter.myo_avg, valid_meter.lv_avg]
    pathology_dice = [valid_meter.scar_avg, valid_meter.edema_avg]

    logger(epoch + 1, args.end_epoch, log_name, cardiac_dice, pathology_dice)

    writer.add_scalar(curve_name_scar, valid_meter.scar_avg.cpu(), epoch)
    writer.add_scalar(curve_name_edema, valid_meter.edema_avg.cpu(), epoch)

    avg_dice = (valid_meter.scar_avg + valid_meter.edema_avg) / 2

    return avg_dice
