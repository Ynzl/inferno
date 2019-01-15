import torch
from .base import Metric
from ...utils.torch_utils import flatten_samples, is_label_tensor
from ...utils.exceptions import assert_, DTypeError, ShapeError


class CategoricalError(Metric):
    """Categorical error."""
    def __init__(self, aggregation_mode='mean'):
        assert aggregation_mode in ['mean', 'sum']
        self.aggregation_mode = aggregation_mode

    def forward(self, prediction, target):
        # Check if prediction is binary or not
        is_binary = len(prediction.size()) == 1 or prediction.size(1) == 1

        if len(target.size()) > 1:
            target = target.squeeze(1)
        assert len(target.size()) == 1

        if is_binary:
            # Binary classification
            prediction = prediction > 0.5
            incorrect = prediction.type_as(target).ne(target).float()
            if self.aggregation_mode == 'mean':
                return incorrect.mean()
            else:
                return incorrect.sum()
        else:
            # Multiclass classificiation
            _, predicted_class = torch.max(prediction, 1)
            if predicted_class.dim() == prediction.dim():
                # Support for Pytorch 0.1.12
                predicted_class = predicted_class.squeeze(1)
            incorrect = predicted_class.type_as(target).ne(target).float()
            if self.aggregation_mode == 'mean':
                return incorrect.mean()
            else:
                return incorrect.sum()


class IOU(Metric):
    """Intersection over Union. """
    def __init__(self, ignore_index=None, ignore_class=None, sharpen_prediction=False, eps=1e-6):
        super(IOU, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.ignore_class = ignore_class
        self.sharpen_prediction = sharpen_prediction

    def forward(self, prediction, target):
        # Assume that is one of:
        #   prediction.shape = (N, C, H, W)
        #   prediction.shape = (N, C, D, H, W)
        #   prediction.shape = (N, C)
        # The corresponding target shapes are either:
        #   target.shape = (N, H, W)
        #   target.shape = (N, D, H, W)
        #   target.shape = (N,)
        # Or:
        #   target.shape = (N, C, H, W)
        #   target.shape = (N, C, D, H, W)
        #   target.shape = (N, C)
        # First, reshape prediction to (C, -1)
        flattened_prediction = flatten_samples(prediction)
        # Take measurements
        num_classes, num_samples = flattened_prediction.size()
        # We need to figure out if the target is a int label tensor or a onehot tensor.
        # The former always has one dimension less, so
        if target.dim() == (prediction.dim() - 1):
            # Labels, we need to go one hot
            # Make sure it's a label
            assert_(is_label_tensor(target),
                    "Target must be a label tensor (of dtype long) if it has one "
                    "dimension less than the prediction.",
                    DTypeError)
            num_labels = max(target.max().item() + 1, num_classes)
            assert_(num_labels <= num_classes + 1,
                    f"too many labels {num_labels}",
                    ValueError)
            # Reshape target to (1, -1) for it to work with scatter
            flattened_target = target.view(1, -1)
            # Convert target to onehot with shape (C, -1)
            onehot_targets = flattened_prediction \
                .new(num_labels, num_samples) \
                .zero_() \
                .scatter_(0, flattened_target, 1)
            # If we are ignoring a label index, remove it from onehot_target
            if self.ignore_index is not None and self.ignore_index < num_labels:
                ignore_index = self.ignore_index \
                    if self.ignore_index != -1 else onehot_targets.size(0) - 1
                dont_ignore_index = list(range(onehot_targets.size(0)))
                dont_ignore_index.pop(ignore_index)
                onehot_targets = onehot_targets[dont_ignore_index]
            # Make sure the target is consistent
            assert_(onehot_targets.size(0) == num_classes)
        elif target.dim() == prediction.dim():
            # Onehot, nothing to do except flatten
            onehot_targets = flatten_samples(target)
        else:
            raise ShapeError("Target must have the same number of dimensions as the "
                             "prediction, or one less. Got target.dim() = {} but "
                             "prediction.dim() = {}.".format(target.dim(), prediction.dim()))
        # Cast onehot_targets to float if required (this is a no-op if it's already float)
        onehot_targets = onehot_targets.float()
        # Sharpen prediction if required to. Sharpening in this sense means to replace
        # the max predicted probability with 1.
        if self.sharpen_prediction:
            _, predicted_classes = torch.max(flattened_prediction, 0)
            # Case for pytorch 0.2, where predicted_classes is (N,) instead of (1, N)
            if predicted_classes.dim() == 1:
                predicted_classes = predicted_classes.view(1, -1)
            # Scatter
            flattened_prediction = flattened_prediction\
                .new(num_classes, num_samples).zero_().scatter_(0, predicted_classes, 1)
        # Now to compute the IOU = (a * b).sum()/(a**2 + b**2 - a * b).sum()
        # We sum over all samples to obtain a classwise iou
        numerator = (flattened_prediction * onehot_targets).sum(-1)
        denominator = \
            flattened_prediction.sub_(onehot_targets).pow_(2).sum(-1) + \
            numerator
        # classwise_iou = numerator.div_(denominator).clamp_(min=self.eps)
        classwise_iou = numerator.div_(denominator)
        # If we're ignoring a class, don't count its contribution to the mean
        if self.ignore_class is not None:
            ignore_class = self.ignore_class \
                if self.ignore_class != -1 else onehot_targets.size(0) - 1
            assert_(ignore_class < onehot_targets.size(0),
                    "`ignore_class` = {} must be at least one less than the number "
                    "of classes = {}.".format(ignore_class, onehot_targets.size(0)),
                    ValueError)
            num_classes = onehot_targets.size(0)
            dont_ignore_class = list(range(num_classes))
            dont_ignore_class.pop(ignore_class)
            if classwise_iou.is_cuda:
                dont_ignore_class = \
                    torch.LongTensor(dont_ignore_class).cuda(classwise_iou.get_device())
            else:
                dont_ignore_class = torch.LongTensor(dont_ignore_class)
            iou = classwise_iou[dont_ignore_class].mean()
        else:
            iou = classwise_iou[~torch.isnan(classwise_iou)].mean()
        return iou


class newIOU(Metric):
    """Intersection over Union.

    ignore_label: index in ground truth that should be ignored (e.g. void label)
    ignore_class: index of class that should be ignored (e.g. background)
    """
    def __init__(self, num_classes, ignore_label=None, ignore_class=None, eps=1e-6):
        super(newIOU, self).__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.ignore_class = ignore_class

    def forward(self, prediction, target):
        flattened_prediction = flatten_samples(prediction).argmax(dim=0)
        # We need to figure out if the target is a int label tensor or a onehot tensor.
        # The former always has one dimension less, so
        if target.dim() == (prediction.dim() - 1):
            # Labels
            # Make sure it's a label
            assert_(is_label_tensor(target),
                    "Target must be a label tensor (of dtype long) if it has one "
                    "dimension less than the prediction.",
                    DTypeError)
            flattened_target = flatten_samples(target)
        elif target.dim() == prediction.dim():
            # Onehot, change to labels
            flattened_target = flatten_samples(target).argmax(dim=1)

        labels = torch.unique(torch.cat((flattened_target, flattened_prediction), dim=0))

        mask = (labels >= 0) & (labels < num_classes)
        confusion = torch.bincount(
                self.num_classes * flattened_target[mask] + flattened_prediction[mask],
                minlength=self.num_classes**2
                ).reshape((self.num_classes, self.num_classes))
        class_iou = torch.diag(confusion) / (confusion.sum(dim=1) + confusion.sum(dim=0) - torch.diag(confusion))

        if self.ignore_class is not None:
            ignore_class = self.ignore_class \
                if self.ignore_class != -1 else num_classes - 1
            dont_ignore_class = list(range(num_classes))
            dont_ignore_class.pop(ignore_class)
            if classwise_iou.is_cuda:
                dont_ignore_class = \
                    torch.LongTensor(dont_ignore_class).cuda(classwise_iou.get_device())
            else:
                dont_ignore_class = torch.LongTensor(dont_ignore_class)
            iou = class_iou[dont_ignore_class].mean()
        else:
            iou = class_iou[~torch.isnan(class_iou)].mean()

        return iou


class NegativeIOU(IOU):
    def forward(self, prediction, target):
        return -1 * super(NegativeIOU, self).forward(prediction, target)
