from .batch_filter import BatchFilter


class MultiplyPredAffsWithMask(BatchFilter):

    def __init__(self, mask_volume_type=None, pred_affs_volume_type=None):
        self.mask_volume_type_to_multiply = mask_volume_type
        self.pred_affs_volume_type = pred_affs_volume_type

    def process(self, batch, request):

        if self.mask_volume_type_to_multiply is None or self.pred_affs_volume_type is None:
            return

        assert batch.volumes[self.mask_volume_type_to_multiply].data.shape == batch.volumes[self.pred_affs_volume_type].data.shape[1:]
        for aff_vector in range(batch.volumes[self.pred_affs_volume_type].data.shape[0]):
            batch.volumes[self.pred_affs_volume_type].data[aff_vector,:,:,:] *= batch.volumes[self.mask_volume_type_to_multiply].data