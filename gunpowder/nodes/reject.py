import logging
import random

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class Reject(BatchFilter):
    '''Reject batches based on the masked-in vs. masked-out ratio.

    Args:

        min_masked(float, optional): The minimal required ratio of masked-in
            vs. masked-out voxels. Defaults to 0.5.

        mask_volume_type(:class:``VolumeTypes``): The mask to use.

        reject_probability(float, optional): The probability by which a batch
            that is not valid (less than min_masked) is actually rejected.
            Defaults to 1., i.e. strict rejection.
    '''

    def __init__(self, min_masked=0.5, mask_volume_type=VolumeTypes.GT_MASK, reject_probability=1.):
        self.min_masked = min_masked
        self.mask_volume_type = mask_volume_type
        self.reject_probability = reject_probability

    def setup(self):
        assert self.mask_volume_type in self.spec, "Reject can only be used if %s is provided"%self.mask_volume_type
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        assert self.mask_volume_type in request, "Reject can only be used if a GT mask is requested"

        have_good_batch = False
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)
            mask_ratio = batch.volumes[self.mask_volume_type].data.mean()
            have_good_batch = mask_ratio>=self.min_masked

            if not have_good_batch and self.reject_probability < 1.:
                have_good_batch = random.random() > self.reject_probability
                logger.debug(
                    "accepted batch with mask ratio %f at" %mask_ratio +
                    str(batch.volumes[self.mask_volume_type].spec.roi))

            if not have_good_batch:

                logger.warning(
                    "reject batch with mask ratio %f at "%mask_ratio +
                    str(batch.volumes[self.mask_volume_type].spec.roi))
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning("rejected %d batches, been waiting for a good one since %ds"%(num_rejected, report_next_timeout))
                    report_next_timeout *= 2

        logger.debug(
            "good batch with mask ratio %f found at "%mask_ratio +
            str(batch.volumes[self.mask_volume_type].spec.roi))

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
