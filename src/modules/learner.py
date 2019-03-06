import torch
from tqdm import tqdm


def get_model(model, checkpoint=None, map_location=None, devices=None):
    model.cuda()

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location) #.module.state_dict()
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd}
        print('Overlapped keys: {}'.format(len(sd.keys())))
        msd.update(sd)
        model.load_state_dict(msd)

    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices)

    return model


def freeze(model, unfreeze=False):
    children = list(model.children())
    if hasattr(model, 'children') and len(children):
        for child in children:
            freeze(child, unfreeze)
    elif hasattr(model, 'parameters'):
        for param in model.parameters():
            param.requires_grad = unfreeze
            
            
def unfreeze_bn(model):
    predicat = isinstance(model, torch.nn.BatchNorm2d)
    predicat |= isinstance(model, bn.ABN)
    predicat |= isinstance(model, bn.InPlaceABN)
    predicat |= isinstance(model, bn.InPlaceABNSync)
    if predicat:
        for param in model.parameters():
            param.requires_grad = True

    children = list(model.children())
    if len(children):
        for child in children:
            unfreeze_bn(child)
    return None


class RetinaLearner:
    def __init__(self, model, opt=None, gclip=0.001):
        self.gclip = gclip
        self.threshold = 0.05

        self.model = model
        self.opt = opt

        if self.opt is not None:
            for group in self.opt.param_groups:
                group.setdefault('initial_lr', group['lr'])

    def make_step(self, data, training=False):
        formated = self._format_input(data)
        prediction = self.model(*formated)
        results = self._format_output(prediction, data)
        
        if training and bool(results['loss'] == 0):
            return results

        if training:
            results['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gclip)
            self.opt.step()

        formated = formated[0][0].data.cpu()
        for key in results.keys():
            results[key] = results[key].data.cpu()
        results.update({ 'pid': data['pid'] })
        return results

    def validate(self, datagen):
        torch.cuda.empty_cache()
        self.model.eval()
        meters = list()
        with torch.no_grad():
            for data in tqdm(datagen.dataset):
                meters.append(self.make_step(data, training=False))
        return meters

    def train_on_epoch(self, datagen, hard_negative_miner=None, lr_scheduler=None):
        torch.cuda.empty_cache()
        self.model.train()
        meters = list()

        for data in tqdm(datagen):
            meters.append(self.make_step(data, training=True))
            if lr_scheduler is not None:
                if hasattr(lr_scheduler, 'batch_step'):
                    lr_scheduler.batch_step(logs=meters[-1])

            if hard_negative_miner is not None:
                hard_negative_miner.update_cache(meters[-1], data)
                if hard_negative_miner.need_iter():
                    self.make_step(hard_negative_miner.get_cache(), training=True)
                    hard_negative_miner.invalidate_cache()

        return meters

    def save(self, path):
        state_dict = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)
        print('Saved in {}:'.format(path))

    def _format_input(self, data):
        image = torch.autograd.Variable(torch.tensor(data['image']))
        if 'bboxes' in data:
            annot = torch.autograd.Variable(torch.tensor(data['bboxes']))

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0).cuda().float()
            if 'bboxes' in data:
                annot = annot.unsqueeze(dim=0).cuda().float()

        is_test = 'bboxes' not in data
        return ([image, annot if not is_test else None], is_test)

    def _format_output(self, prediction, data):
        results = dict()
        if 'focal_loss' in prediction:
            classification_loss, regression_loss = prediction['focal_loss']
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            results = { 
                'loss': classification_loss + regression_loss,
                'bbx_reg_loss': regression_loss,
                'bbx_clf_loss': classification_loss,
            }
        if 'nms_out' in prediction:
            results.update(prediction['nms_out'])
        if 'bboxes' in data:
            results.update({ 'annotation': data['bboxes'] })
        return results

